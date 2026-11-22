"""Generate FLUX Table 1 images using newly calibrated permutations."""

import argparse
import json
import time
from pathlib import Path

from common import BASE, flux_generate, get_noise, load_ae, load_flow_model, prepare
from common import quant_config, read_json, seed, sha256, source_integrity, torch, write_json
from fake_quant.quant.reorder_quant2 import quant_flux_reorder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["plain", "hadamard"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Per evaluation subset; 0 means full set")
    parser.add_argument("--offload-decode", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    embeddings = args.embeddings.resolve()
    checkpoint = args.checkpoint.resolve()
    indices = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if len(indices) != 228:
        raise ValueError(f"Expected a full 228-layer permutation checkpoint, got {len(indices)}")
    all_tasks = read_json(BASE / "tasks.json")
    tasks = []
    for subset in ["geneval", "mjhq30k"]:
        chosen = [task for task in all_tasks if task["subset"] == subset]
        tasks.extend(chosen[: args.limit or None])
    config = quant_config(args.variant)
    write_json(
        output / "run_manifest.json",
        {
            "variant": args.variant,
            "config": config,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "tasks": tasks,
            "seed_per_image": 42,
            "height": 1024,
            "width": 1024,
            "steps": 50,
            "guidance": 3.5,
            "dtype": "bfloat16",
            "embedding_cache": str(embeddings),
            "offload_decode": args.offload_decode,
            "source_sha256": source_integrity(),
        },
    )
    seed()
    with torch.no_grad():
        model = load_flow_model("flux-dev", device="cuda").eval()
        quant_flux_reorder(model, config, reorder_cache_path=str(checkpoint))
        torch.cuda.empty_cache()
        ae = load_ae("flux-dev", device="cpu" if args.offload_decode else "cuda").eval()
        for index, task in enumerate(tasks):
            directory = output / task["subset"] / "images" / task["id"]
            image_path = directory / "samples" / f"{task['id']}.png"
            if image_path.exists():
                metadata_path = directory / "metadata.jsonl"
                if not metadata_path.exists():
                    metadata_path.write_text(json.dumps(task["metadata"]) + "\n")
                continue
            encoded = torch.load(
                embeddings / task["subset"] / f"{task['id']}.pt", map_location="cpu", weights_only=True
            )
            prompt = task["metadata"]["prompt"]
            if encoded["prompt"] != prompt:
                raise ValueError(f"Prompt mismatch: {task['subset']} {task['id']}")
            seed()
            start = time.time()
            noise = get_noise(1, 1024, 1024, device=torch.device("cuda"), dtype=torch.bfloat16, seed=42)
            # prepare is the original function. Adapters return the saved exact
            # T5/CLIP outputs instead of loading the large encoders repeatedly.
            inputs = prepare(lambda _: encoded["txt"], lambda _: encoded["vec"], noise, prompt=prompt)
            image = flux_generate(
                model, ae, inputs, 1024, 1024, 50, 3.5, torch.device("cuda"), offload=args.offload_decode
            )
            image_path.parent.mkdir(parents=True, exist_ok=True)
            temp = image_path.with_suffix(".tmp.png")
            image.save(temp)
            temp.replace(image_path)
            # MJHQ loader reads one JSON line, whereas GenEval accepts JSON.
            (directory / "metadata.jsonl").write_text(json.dumps(task["metadata"]) + "\n")
            if args.offload_decode:
                ae.decoder.cpu()
                torch.cuda.empty_cache()
            print(
                json.dumps(
                    {
                        "completed": index + 1,
                        "total": len(tasks),
                        "subset": task["subset"],
                        "id": task["id"],
                        "seconds": time.time() - start,
                        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                    }
                ),
                flush=True,
            )
    write_json(output / "generation_complete.json", {"images": len(tasks)})


if __name__ == "__main__":
    main()
