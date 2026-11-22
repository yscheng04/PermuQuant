import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import functools
import json
import random
import shutil

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from app.sana_pipeline import SanaPipeline

from fake_quant.collect.utils import CollectHook, process, tree_map
from fake_quant.dataset.calib_dataset import CalibDataset


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(args):
    sana = SanaPipeline(args.config)
    sana.from_pretrained(args.model_path)
    return {
        "pipeline": sana,
        "model": sana.model,
        "device": sana.device,
    }


def get_dataset(args):
    dataset_type = {
        "calib": CalibDataset,
    }
    dataset_cls = dataset_type[args.dataset_name]
    num_samples_gen = args.num_samples_gen if args.num_samples_gen >= 0 else 0
    return dataset_cls(
        prompt_file=args.prompt_file,
        num_samples_gen=num_samples_gen,
        seed=args.seed,
    )


def wrap_forward_for_collection(model, caches):
    """Capture top-level forward inputs/outputs even when callers bypass nn.Module.__call__."""
    collector = CollectHook(caches)
    original_forward = model.forward

    @functools.wraps(original_forward)
    def wrapped_forward(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        collector(model, args, kwargs, output)
        return output

    model.forward = wrapped_forward


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    sample = image_tensor.detach().cpu().float()
    sample = sample.mul(0.5).add(0.5).clamp(0, 1)
    ndarr = sample.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
    return Image.fromarray(ndarr)


@torch.inference_mode()
def collect(args):
    seed_everything(args.seed)

    components = get_model(args)
    pipeline = components["pipeline"]
    model = components["model"]
    device = components["device"]

    dataset = get_dataset(args)

    sample_dirpath = os.path.join(args.save_path, "sample")
    cache_dirpath = os.path.join(args.save_path, "cache")
    os.makedirs(args.save_path, exist_ok=True)
    if os.path.exists(sample_dirpath):
        shutil.rmtree(sample_dirpath)
    if os.path.exists(cache_dirpath):
        shutil.rmtree(cache_dirpath)
    os.makedirs(sample_dirpath, exist_ok=True)
    os.makedirs(cache_dirpath, exist_ok=True)
    manifest_path = os.path.join(args.save_path, "prompts.json")
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(
            {
                "prompt_file": os.path.abspath(args.prompt_file),
                "seed": args.seed,
                "num_prompts": len(dataset),
                "num_inference_steps": args.num_inference_steps,
                "prompts": dataset.prompts,
            },
            manifest_file,
            ensure_ascii=False,
            indent=2,
        )
        manifest_file.write("\n")

    caches = []
    wrap_forward_for_collection(model, caches)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    log_path = os.path.join(sample_dirpath, "log.txt")
    log_file = open(log_path, "w", encoding="utf-8")

    print(f"In total {len(dataset)} samples")

    for data in tqdm(dataset, desc="Data", leave=False, dynamic_ncols=True):
        file_name = data["file_name"]
        prompt = data["prompt"]
        mode = data.get("mode", "Gen")

        image = pipeline(
            prompt=prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            pag_guidance_scale=args.pag_guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        )
        output_image = tensor_to_pil(image[0])

        log_file.write(f"Task:{mode} Filename:{file_name} Prompt:{prompt}\n")
        output_image.save(os.path.join(sample_dirpath, file_name))

        for idx, c in enumerate(caches):
            c = tree_map(lambda x: process(x), c)
            cache_path = os.path.join(cache_dirpath, f"{os.path.splitext(file_name)[0]}_{idx:03d}.pt")
            torch.save(c, cache_path)

        caches.clear()
        log_file.flush()

    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sana1-5_config/1024ms/Sana_1600M_1024px_allqknorm_bf16_lr2e5.yaml",
        help="Path to the Sana config",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="hf://Efficient-Large-Model/SANA1.5_1.6B_1024px/checkpoints/SANA1.5_1.6B_1024px.pth",
        help="Path to the model checkpoint",
    )
    parser.add_argument("--dataset_name", type=str, default="calib", choices=["calib"], help="Name of the dataset to use")
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="eval/geneval/prompts/evaluation_metadata_long.jsonl",
        help="Path to the prompt source",
    )
    parser.add_argument("--save_path", type=str, default="fake_quant/collect/output", help="Directory to save collected data")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=4.5, help="Guidance scale for CFG")
    parser.add_argument("--pag_guidance_scale", type=float, default=1.0, help="PAG guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_samples_gen", type=int, default=4, help="Number of prompt samples to process from the dataset, -1 for all")
    parser.add_argument("--num_samples", type=int, default=None, help="Deprecated alias for --num_samples_gen")

    args = parser.parse_args()
    if args.num_samples is not None:
        args.num_samples_gen = args.num_samples
    collect(args)
