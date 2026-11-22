import os
import sys

# Add src/ to path for flux imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))

from PIL import Image
import torch
from torch import nn
import random
import numpy as np
from tqdm import tqdm
import shutil
import argparse


def print_gpu_memory(tag=""):
    """Print GPU memory usage statistics."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n[GPU Memory] {tag}")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
        print(f"  Peak:      {max_allocated:.2f} GB")
        print(f"  Free:      {torch.cuda.get_device_properties(0).total_memory / 1024**3 - reserved:.2f} GB")

from fake_quant.dataset.calib_dataset import CalibDataset
from fake_quant.collect.utils import CollectHook
from fake_quant.utils import tree_map

from einops import rearrange
from flux.sampling import prepare, denoise, get_schedule, unpack, get_noise
from flux.util import load_flow_model, load_t5, load_clip, load_ae, save_image, embed_watermark


def process(x: torch.Tensor) -> torch.Tensor:
    """Process tensor to ensure CPU compatibility."""
    dtype = x.dtype
    return torch.from_numpy(x.float().numpy()).to(dtype)


def get_dataset(args):
    """Get calibration dataset."""
    dataset_name = "calib"
    DatasetType = {"calib": CalibDataset}

    dataset = DatasetType[dataset_name](
        prompt_folder=args.dataset_path,
        num_samples=args.num_samples
    )
    return dataset


@torch.inference_mode()
def collect(args):
    """Main collection function for FLUX model calibration data."""
    # Set seeds for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Configuration
    dtype = torch.bfloat16
    height = args.height
    width = args.width
    num_inference_steps = args.num_steps
    guidance_scale = args.guidance
    name = args.name

    # Device selection
    device = torch.device(args.device)
    print(f"Chosen device: {device}")

    # Load models
    print(f"Loading model: {name}")
    if num_inference_steps is None:
        num_inference_steps = 4 if name == "flux-schnell" else 50

    # Adjust guidance for schnell
    if name == "flux-schnell":
        guidance_scale = 0.0

    t5 = load_t5(device, max_length=256 if name == "flux-schnell" else 512)
    clip = load_clip(device)
    model = load_flow_model(name, device="cpu" if args.offload else device)
    ae = load_ae(name, device="cpu" if args.offload else device)

    # Get dataset
    dataset = get_dataset(args)

    # Setup output directories
    sample_dirpath = os.path.join(args.save_path, "sample")
    cache_dirpath = os.path.join(args.save_path, "cache")

    # Clear directories if they exist
    if os.path.exists(sample_dirpath):
        shutil.rmtree(sample_dirpath)
    if os.path.exists(cache_dirpath):
        shutil.rmtree(cache_dirpath)
    os.makedirs(sample_dirpath, exist_ok=True)
    os.makedirs(cache_dirpath, exist_ok=True)

    caches = []

    # Register hook on the transformer model
    assert isinstance(model, nn.Module), "The model must be an instance of nn.Module"
    model.register_forward_hook(CollectHook(caches), with_kwargs=True)

    print(f"In total {len(dataset)} samples")

    # Open log file
    log_path = os.path.join(sample_dirpath, "log.txt")
    log_file = open(log_path, "w", encoding="utf-8")

    # Process each sample
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        file_name = data['file_name']
        prompt = data['prompt']

        # Prepare inputs
        x = get_noise(
            1,
            height,
            width,
            device=device,
            dtype=dtype,
            seed=seed,
        )

        # Offload handling
        if args.offload:
            t5, clip = t5.to(device), clip.to(device)

        inp = prepare(t5, clip, x, prompt=prompt)
        timesteps = get_schedule(
            num_inference_steps,
            inp["img"].shape[1],
            shift=(name != "flux-schnell")
        )

        # Offload TEs to CPU, load model to GPU
        if args.offload:
            t5, clip = t5.cpu(), clip.cpu()
            torch.cuda.empty_cache()
            model = model.to(device)

        # Denoise (Hook will capture activations)
        x = denoise(model, **inp, timesteps=timesteps, guidance=guidance_scale)

        # Offload model, load autoencoder to GPU
        if args.offload:
            model = model.cpu()
            torch.cuda.empty_cache()
            ae.decoder.to(x.device)

        # Decode latents to pixel space
        x = unpack(x.float(), height, width)
        with torch.autocast(device_type=device.type, dtype=dtype):
            x = ae.decode(x)

        # Save image (same logic as cli.py save_image function)
        x = x.clamp(-1, 1)
        x = embed_watermark(x.float())
        x = rearrange(x[0], "c h w -> h w c")
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        log_file.write(f"Filename:{file_name} Prompt:{prompt}\n")
        img.save(os.path.join(sample_dirpath, file_name))

        # Save cached activations
        for idx, c in enumerate(caches):
            c = tree_map(lambda x: process(x), c)
            cache_path = os.path.join(cache_dirpath, f"{os.path.splitext(file_name)[0]}_{idx:03d}.pt")
            torch.save(c, cache_path)

        log_file.flush()
        caches.clear()

    log_file.close()
    print(f"Collection complete. Samples saved to {sample_dirpath}, caches saved to {cache_dirpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect calibration data for FLUX model quantization")

    # Model parameters
    parser.add_argument("--name", type=str, default="flux-dev",
                        help="Model name (flux-dev, flux-schnell, etc.)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda, cpu)")
    parser.add_argument("--offload", action="store_true",
                        help="Offload models to CPU when not in use")

    # Generation parameters
    parser.add_argument("--height", type=int, default=1024,
                        help="Image height (must be multiple of 16)")
    parser.add_argument("--width", type=int, default=1024,
                        help="Image width (must be multiple of 16)")
    parser.add_argument("--num_steps", type=int, default=None,
                        help="Number of inference steps (default: 4 for schnell, 50 for dev)")
    parser.add_argument("--guidance", type=float, default=2.5,
                        help="Guidance scale (ignored for schnell)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Dataset parameters
    parser.add_argument("--dataset_path", type=str,
                        default="data/coco-caption/coco_karpathy_test.json",
                        help="Path to COCO caption dataset")
    parser.add_argument("--num_samples", type=int, default=64,
                        help="Number of prompt samples to process from the dataset")

    # Output parameters
    parser.add_argument("--save_path", type=str, default="fake_quant/collect/output",
                        help="Directory to save collected data")

    args = parser.parse_args()

    # Ensure height and width are multiples of 16
    args.height = 16 * (args.height // 16)
    args.width = 16 * (args.width // 16)

    collect(args)
