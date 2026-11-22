"""Recollect the 64 archived COCO prompts with the original FLUX FP pipeline."""

import argparse
from pathlib import Path

from common import BASE, encoders, flux_generate, get_noise, load_ae, load_flow_model
from common import prepare, read_json, seed, source_integrity, torch, write_json
from fake_quant.collect.collect import process
from fake_quant.collect.utils import CollectHook
from fake_quant.utils import tree_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prompts = read_json(BASE / "calibration_prompts.json")[: args.limit]
    source_hashes = source_integrity()
    write_json(
        output / "collection_manifest.json",
        {
            "prompts": prompts,
            "seed": 42,
            "height": 1024,
            "width": 1024,
            "steps": 50,
            "guidance": 2.5,
            "guidance_provenance": "Original collector default; historical collection parameters not fully logged",
            "main_device": "cuda:0",
            "aux_device": "cuda:1",
            "source_sha256": source_hashes,
        },
    )
    seed()
    with torch.no_grad():
        t5, clip = encoders("cuda:1")
        model = load_flow_model("flux-dev", device="cuda:0").eval()
        ae = load_ae("flux-dev", device="cuda:1").eval()
        caches = []
        model.register_forward_hook(CollectHook(caches), with_kwargs=True)
        for index, prompt in enumerate(prompts):
            name = f"{index:03d}"
            files = [output / "cache" / f"{name}_{step:03d}.pt" for step in range(50)]
            if all(f.exists() for f in files) and (output / "sample" / f"{name}.png").exists():
                continue
            caches.clear()
            seed()
            noise = get_noise(1, 1024, 1024, device=torch.device("cuda:0"), dtype=torch.bfloat16, seed=42)
            inputs = prepare(t5, clip, noise, prompt=prompt)
            image = flux_generate(model, ae, inputs, 1024, 1024, 50, 2.5, torch.device("cuda:0"))
            assert len(caches) == 50, len(caches)
            for path, cache in zip(files, caches):
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix(".tmp")
                torch.save(tree_map(process, cache), temp)
                temp.replace(path)
            (output / "sample").mkdir(parents=True, exist_ok=True)
            image.save(output / "sample" / f"{name}.png")
            caches.clear()
            print(f"COLLECTED {index + 1}/{len(prompts)}: 50 forward inputs", flush=True)
    (output / "sample/log.txt").write_text(
        "".join(f"Filename:{i:03d}.png Prompt:{p}\n" for i, p in enumerate(prompts))
    )
    write_json(
        output / "collection_complete.json", {"prompts": len(prompts), "forward_inputs": len(prompts) * 50}
    )


if __name__ == "__main__":
    main()
