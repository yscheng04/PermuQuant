"""Recompute FLUX permutations with bounded-memory replay of the original loader."""

import argparse
from pathlib import Path

from common import load_flow_model, quant_config, seed, sha256, source_integrity, torch, write_json
from fake_quant.calib.calib_cache_loader import CalibCacheLoader
import fake_quant.quant.reorder_quant2 as reorder_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["plain", "hadamard"], required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    inputs = args.inputs.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "reorder.pt"
    if cache.exists():
        raise FileExistsError(f"Refusing to overwrite {cache}")
    config = quant_config(args.variant)

    class ReplayLoader(CalibCacheLoader):
        def __init__(self, *a, **kw):
            kw.update(sample_path=str(inputs))
            if args.smoke:
                kw["max_samples"] = 5
            super().__init__(*a, **kw)

        def iter_layer_activations(self, model, *a, **kw):
            blocks = list(model.double_blocks) + list(model.single_blocks)
            if args.smoke:
                blocks = blocks[:1]
            for block in blocks:
                # Both arguments are supported by the existing loader. Stopping
                # after this block avoids computing irrelevant later blocks.
                yield from super().iter_layer_activations(
                    model, *a, layers=[block], early_stop_module=block, **kw
                )

    reorder_module.CalibCacheLoader = ReplayLoader
    loader = ReplayLoader(sample_ratio=config["sample_ratio"])
    files = loader.dataset.files
    write_json(
        output / "calibration_manifest.json",
        {
            "variant": args.variant,
            "config": config,
            "smoke_only": args.smoke,
            "selected_forward_inputs": [
                {"file": str(Path(f).relative_to(inputs)), "sha256": sha256(f)} for f in files
            ],
            "input_count": len(files),
            "replay": "original loader, one block at a time, early_stop_module=current block",
            "source_sha256": source_integrity(),
        },
    )
    if not args.smoke:
        assert len(files) == 320, f"Expected 320 selected inputs, found {len(files)}"
    seed()
    model = load_flow_model("flux-dev", device="cuda").eval()
    reorder_module.quant_flux_reorder(model, config, reorder_cache_path=str(cache))
    tensors = torch.load(cache, map_location="cpu", weights_only=False)
    expected = 8 if args.smoke else 228
    assert len(tensors) == expected, (len(tensors), expected)
    write_json(
        output / "calibration_complete.json",
        {
            "variant": args.variant,
            "smoke_only": args.smoke,
            "layers": len(tensors),
            "accepted": sum(t is not None for t in tensors.values()),
            "sha256": sha256(cache),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        },
    )


if __name__ == "__main__":
    main()
