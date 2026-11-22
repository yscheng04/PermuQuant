import os

from PIL import Image
import torch
from torch import nn
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
import random
import numpy as np
from tqdm import tqdm
import shutil
import argparse

from fake_quant.dataset.calib_dataset import CalibDataset
from fake_quant.collect.utils import CollectHook
from fake_quant.utils import tree_map
from utils import AttentionBackend, ensure_model_weights, load_from_local_dir, set_attention_backend
from zimage import generate

def process(x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    return torch.from_numpy(x.float().numpy()).to(dtype)

def get_dataset(args):
    dataset_name = "calib"
    DatasetType = {"calib": CalibDataset,}

    dataset = DatasetType[dataset_name](num_samples=args.num_samples)
    return dataset

def collect(args):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_path = ensure_model_weights("ckpts/Z-Image-Turbo", verify=False)  # True to verify with md5
    dtype = torch.bfloat16
    compile = False  # default False for compatibility
    height = 1024
    width = 1024
    num_inference_steps = 8
    guidance_scale = 0.0
    attn_backend = os.environ.get("ZIMAGE_ATTENTION", "_native_flash")

    # Device selection priority: cuda -> tpu -> mps -> cpu
    if torch.cuda.is_available():
        device = "cuda"
        print("Chosen device: cuda")
    else:
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm

            device = xm.xla_device()
            print("Chosen device: tpu")
        except (ImportError, RuntimeError):
            if torch.backends.mps.is_available():
                device = "mps"
                print("Chosen device: mps")
            else:
                device = "cpu"
                print("Chosen device: cpu")
    # Load models
    components = load_from_local_dir(model_path, device=device, dtype=dtype, compile=compile)
    AttentionBackend.print_available_backends()
    set_attention_backend(attn_backend)
    print(f"Chosen attention backend: {attn_backend}")

    dataset = get_dataset(args)

    sample_dirpath = os.path.join(args.save_path, "sample")
    cache_dirpath = os.path.join(args.save_path, "cache")
    # 每次跑之前清空文件夹
    if os.path.exists(sample_dirpath):
        shutil.rmtree(sample_dirpath)
    if os.path.exists(cache_dirpath):
        shutil.rmtree(cache_dirpath)
    os.makedirs(sample_dirpath, exist_ok=True)
    os.makedirs(cache_dirpath, exist_ok=True)
    caches = []

    llm_backbone = components['transformer']
    assert isinstance(llm_backbone, nn.Module), "The language model must be an instance of nn.Module"
    llm_backbone.register_forward_hook(CollectHook(caches), with_kwargs=True)

    print(f"In total {len(dataset)} samples")

    log_path = os.path.join(sample_dirpath, "log.txt")
    log_file = open(log_path, "w", encoding="utf-8")

    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        file_name = data['file_name']
        prompt = data['prompt']

        images = generate(
            prompt=prompt,
            **components,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device).manual_seed(seed),
        )

        log_file.write(f"Filename:{file_name} Prompt:{prompt}\n")
        images[0].save(os.path.join(sample_dirpath, file_name))

        for idx, c in enumerate(caches):
            c = tree_map(lambda x: process(x), c)
            cache_path = os.path.join(cache_dirpath, f"{os.path.splitext(file_name)[0]}_{idx:03d}.pt")
            torch.save(c, cache_path)

        log_file.flush()
        caches.clear()
    
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="fake_quant/collect/output", help="Directory to save collected data")
    parser.add_argument("--num_samples", type=int, default=64, help="Number of prompt samples to process from the dataset")

    args = parser.parse_args()

    collect(args)