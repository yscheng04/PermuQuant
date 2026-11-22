"""Replay archived Sana Table 1 samples through the original pipeline."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
SANA = ROOT / "Sana"
BASE = SANA / "fake_quant/cli"
for key in ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"]:
    os.environ[key] = str(SANA / "cache/huggingface/hub")
os.environ.update(
    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TORCH_HOME=str(SANA / "cache/torch")
)
sys.path.insert(0, str(SANA))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["plain", "hadamard"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    out = args.output.resolve()
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    os.chdir(SANA)
    from runtime import load_sana_components, generate_image
    from fake_quant.quant import quant_sana_reorder
    import torch

    config = json.loads((BASE / (args.variant + ".json")).read_text())
    checkpoint = (
        checkpoint or BASE / "runs" / (args.variant + "-calibration") / "reorder.pt"
    )
    assert checkpoint.is_file(), checkpoint
    config["config"] = str(SANA / config["config"])
    config["model_path"] = str(SANA / "checkpoints/SANA1.5_1.6B_1024px.pth")
    all_tasks = json.loads((BASE / "tasks.json").read_text())
    tasks = [
        t
        for subset in ["geneval", "mjhq30k"]
        for t in [x for x in all_tasks if x["subset"] == subset][: args.limit or None]
    ]
    sources = {
        str(p.relative_to(SANA)): digest(p)
        for folder in ["app", "diffusion", "fake_quant"]
        for p in (SANA / folder).rglob("*.py")
    }
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        variant=args.variant,
        config=config,
        checkpoint=str(checkpoint),
        checkpoint_sha256=digest(checkpoint),
        source_sha256=sources,
        tasks=tasks,
    )
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    components = load_sana_components(argparse.Namespace(**config))
    quant_sana_reorder(
        components["pipeline"].model,
        config["quant_config"],
        reorder_cache_path=str(checkpoint),
    )
    torch.cuda.empty_cache()
    for n, task in enumerate(tasks):
        directory = out / task["subset"] / "images" / task["id"]
        image = directory / "samples" / (task["id"] + ".png")
        if image.exists():
            continue
        start = time.time()
        result = generate_image(
            task["metadata"]["prompt"], components, argparse.Namespace(**config), 42
        )
        image.parent.mkdir(parents=True, exist_ok=True)
        temp = image.with_suffix(".tmp.png")
        result.save(temp)
        temp.replace(image)
        (directory / "metadata.jsonl").write_text(json.dumps(task["metadata"]) + "\n")
        print(
            json.dumps(
                dict(completed=n + 1, total=len(tasks), seconds=time.time() - start)
            ),
            flush=True,
        )
    (out / "generation_complete.json").write_text(json.dumps(dict(images=len(tasks))))


if __name__ == "__main__":
    main()
