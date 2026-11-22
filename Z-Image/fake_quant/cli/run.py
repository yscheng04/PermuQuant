"""Replay archived Table 1 samples using unmodified Z-Image quantization code."""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
Z = ROOT / "Z-Image"
for key in ("HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE"):
    os.environ[key] = str(Z / "cache/huggingface/hub")
os.environ.update(
    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TORCH_HOME=str(Z / "cache/torch")
)
sys.path[:0] = [str(Z / "src"), str(Z)]
os.chdir(Z)
import numpy as np
import torch
from runtime import default_quant_config
from fake_quant.quant.reorder_quant2 import quant_zimage_reorder
from utils import load_from_local_dir, set_attention_backend
from zimage import generate


def seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["plain", "hadamard"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--split-components", action="store_true")
    p.add_argument("--original-grad-context", action="store_true")
    p.add_argument("--reorder-cache", type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = a.reorder_cache.resolve()
    if not cache.is_file():
        raise FileNotFoundError(cache)
    config = default_quant_config()
    config.update(
        smooth=False, svd=False, hadamard=a.variant == "hadamard", calib_ratio=0.2
    )
    sources = {
        str(f.relative_to(Z)): hashlib.sha256(f.read_bytes()).hexdigest()
        for base in ["src", "fake_quant/quant", "fake_quant/calib"]
        for f in (Z / base).rglob("*.py")
    }
    all_tasks = json.loads((Path(__file__).parent / "tasks.json").read_text())
    tasks = [
        (t["subset"], t["id"], t["metadata"])
        for subset in ["geneval", "mjhq30k"]
        for t in [x for x in all_tasks if x["subset"] == subset][: a.limit or None]
    ]
    record = {
        "variant": a.variant,
        "original_grad_context": a.original_grad_context,
        "split_components": a.split_components,
        "quant_config": config,
        "cache": str(cache),
        "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        "source_sha256": sources,
        "seed_per_image": 42,
        "height": 1024,
        "width": 1024,
        "steps": 8,
        "guidance_scale": 0,
        "attention_backend": "_native_flash",
        "tasks": [(s, i, m) for s, i, m in tasks],
    }
    (out / "run_manifest.json").write_text(json.dumps(record, indent=2))
    seed()
    print("Loading model", flush=True)
    components = load_from_local_dir(
        str(Z / "ckpts/Z-Image-Turbo"),
        device="cpu" if a.split_components else "cuda",
        dtype=torch.bfloat16,
        compile=False,
    )
    if a.split_components:
        components["transformer"].to("cuda:0")
        components["text_encoder"].to("cuda:1")
        components["vae"].to("cuda:1")
    set_attention_backend("_native_flash")
    quant_zimage_reorder(
        components["transformer"], config, reorder_cache_path=str(cache)
    )
    torch.cuda.empty_cache()
    print("QUANTIZATION_READY", flush=True)
    for n, (subset, idx, meta) in enumerate(tasks):
        d = out / subset / "images" / idx
        image = d / "samples" / f"{idx}.png"
        if image.exists():
            continue
        image.parent.mkdir(parents=True, exist_ok=True)
        seed()
        start = time.time()
        with nullcontext():
            with torch.no_grad() if a.original_grad_context else torch.inference_mode():
                result = generate(
                    **components,
                    prompt=meta["prompt"],
                    height=1024,
                    width=1024,
                    num_inference_steps=8,
                    guidance_scale=0,
                    generator=torch.Generator("cuda").manual_seed(42),
                )
        temp = image.with_suffix(".tmp.png")
        result[0].save(temp)
        temp.replace(image)
        (d / "metadata.jsonl").write_text(json.dumps(meta) + "\n")
        print(
            json.dumps(
                {
                    "completed": n + 1,
                    "total": len(tasks),
                    "subset": subset,
                    "id": idx,
                    "seconds": time.time() - start,
                    "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
            ),
            flush=True,
        )
    (out / "generation_complete.json").write_text(json.dumps({"images": len(tasks)}))


if __name__ == "__main__":
    main()
