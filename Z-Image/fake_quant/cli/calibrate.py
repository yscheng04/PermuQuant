"""Repeat reorder search on preserved forward inputs without overwriting originals."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
from run import (
    ROOT,
    Z,
    seed,
    default_quant_config,
    load_from_local_dir,
    set_attention_backend,
    quant_zimage_reorder,
    torch,
)
from fake_quant.calib.calib_cache_loader import CalibCacheLoader

p = argparse.ArgumentParser()
p.add_argument("--variant", choices=["plain", "hadamard"], required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--ratio", type=float, default=1.0)
p.add_argument("--forward-cache", type=Path)
p.add_argument("--input-order-manifest", type=Path)
a = p.parse_args()
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=True)
new = out / "reorder.pt"
if new.exists():
    raise SystemExit("Refusing to overwrite an existing recalibration checkpoint")
suffix = "_quarot" if a.variant == "hadamard" else ""
old = (
    Z / "fake_quant/outputs" / f"reorder2_cache_W3A3_group_32_sm{suffix}_threshold_0.pt"
)
cfg = default_quant_config()
cfg.update(
    smooth=False, svd=False, hadamard=a.variant == "hadamard", calib_ratio=a.ratio
)
# Supply paths/order at the wrapper boundary; the original calibration algorithm
# and dataset remain unchanged. Defaults select the same archived input files.
import fake_quant.quant.reorder_quant2 as reorder_module

forward_path = (a.forward_cache or Z / "fake_quant/collect/output/cache").resolve()
order = None
if a.input_order_manifest:
    order = json.loads(a.input_order_manifest.read_text())[
        "all_files_in_shuffled_order"
    ]


def make_loader(*args, **kwargs):
    kwargs.setdefault("sample_path", str(forward_path))
    instance = CalibCacheLoader(*args, **kwargs)
    if order is not None:
        names = [Path(f).name for f in order]
        actual = {Path(f).name for f in instance.dataset.files}
        if len(names) != len(actual) or set(names) != actual:
            raise ValueError("Replay input order does not match the selected cache")
        instance.dataset.files = [str(forward_path / name) for name in names]
    return instance


reorder_module.CalibCacheLoader = make_loader
loader = make_loader(ratio=a.ratio)
files = loader.dataset.files
selected = files[: loader.num_samples]
manifest = {
    "variant": a.variant,
    "quant_config": cfg,
    "forward_cache_origin": "user-supplied cache"
    if a.forward_cache
    else "preserved archive; not recaptured",
    "forward_cache_path": str(forward_path),
    "input_order_replayed": a.input_order_manifest is not None,
    "old_cache": str(old.relative_to(ROOT)),
    "all_files_in_shuffled_order": files,
    "selected_files": [
        {"file": f, "sha256": hashlib.sha256(Path(f).read_bytes()).hexdigest()}
        for f in selected
    ],
}
(out / "calibration_manifest.json").write_text(json.dumps(manifest, indent=2))
seed()
components = load_from_local_dir(
    str(Z / "ckpts/Z-Image-Turbo"), device="cpu", dtype=torch.bfloat16, compile=False
)
model = components["transformer"]
del components
gc.collect()
model.to("cuda")
set_attention_backend("_native_flash")
quant_zimage_reorder(model, cfg, reorder_cache_path=str(new))
y = torch.load(new, map_location="cpu", weights_only=False)
assert len(y) > 0
(out / "calibration_complete.json").write_text(
    json.dumps(
        {"layers": len(y), "sha256": hashlib.sha256(new.read_bytes()).hexdigest()},
        indent=2,
    )
)
