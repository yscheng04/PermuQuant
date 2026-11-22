"""Encode the exact 1200 evaluation prompts once for both FLUX variants."""

import argparse
from pathlib import Path

from common import BASE, encoders, get_noise, prepare, read_json, seed, source_integrity, torch
from common import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tasks = read_json(BASE / "tasks.json")
    write_json(
        output / "encoding_manifest.json",
        {
            "tasks": tasks,
            "source_sha256": source_integrity(),
            "dtype": "bfloat16",
            "text_length": 512,
            "shared_between_variants": True,
        },
    )
    seed()
    with torch.no_grad():
        t5, clip = encoders("cuda")
        noise = get_noise(1, 1024, 1024, device=torch.device("cpu"), dtype=torch.bfloat16, seed=42)
        for index, task in enumerate(tasks):
            path = output / task["subset"] / f"{task['id']}.pt"
            if path.exists():
                data = torch.load(path, map_location="cpu", weights_only=True)
                if data["prompt"] != task["metadata"]["prompt"]:
                    raise ValueError(f"Prompt cache conflict: {path}")
                continue
            seed()
            inputs = prepare(t5, clip, noise, prompt=task["metadata"]["prompt"])
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".tmp")
            torch.save(
                {"prompt": task["metadata"]["prompt"], "txt": inputs["txt"], "vec": inputs["vec"]}, temp
            )
            temp.replace(path)
            print(f"ENCODED {index + 1}/{len(tasks)}", flush=True)
    write_json(output / "encoding_complete.json", {"prompts": len(tasks)})


if __name__ == "__main__":
    main()
