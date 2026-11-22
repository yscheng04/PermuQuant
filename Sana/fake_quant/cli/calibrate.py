"""Call original Sana reorder search on recollected inputs, preserving historical caches."""

import argparse
import json
import os
from pathlib import Path
from run import SANA, BASE, digest

p = argparse.ArgumentParser()
p.add_argument("--variant", choices=["plain", "hadamard"], required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=True)
cache = out / "reorder.pt"
assert not cache.exists(), cache
os.chdir(SANA)
from runtime import load_sana_components
from fake_quant.collect.collect import seed_everything
from fake_quant.quant import quant_sana_reorder
import torch

config = json.loads((BASE / (a.variant + ".json")).read_text())
config["config"] = str(SANA / config["config"])
config["model_path"] = str(SANA / "checkpoints/SANA1.5_1.6B_1024px.pth")
inputs = BASE / "runs/forward-inputs/cache"
q = config["quant_config"]
q["calib_sample_path"] = str(inputs)
seed_everything(42)
components = load_sana_components(argparse.Namespace(**config))
pipe = components["pipeline"]
# Text encoder and VAE are unnecessary for replay of saved model inputs.
pipe.text_encoder.cpu()
pipe.vae.cpu()
torch.cuda.empty_cache()
(out / "calibration_manifest.json").write_text(
    json.dumps(
        dict(
            variant=a.variant,
            config=config,
            inputs=[
                dict(file=f.name, sha256=digest(f)) for f in sorted(inputs.glob("*.pt"))
            ],
        ),
        indent=2,
    )
)
quant_sana_reorder(pipe.model, q, reorder_cache_path=str(cache))
indices = torch.load(cache, map_location="cpu", weights_only=False)
assert len(indices) == 140, len(indices)
(out / "calibration_complete.json").write_text(
    json.dumps(
        dict(
            variant=a.variant,
            layers=len(indices),
            accepted=sum(v is not None for v in indices.values()),
            sha256=digest(cache),
        ),
        indent=2,
    )
)
