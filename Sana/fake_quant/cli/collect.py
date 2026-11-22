"""Recollect archived prompts with original Sana collector and generator semantics."""

import argparse
import json
import os
from pathlib import Path
from run import SANA, BASE, digest

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
out = a.output.resolve()
assert not out.exists(), f"Refusing to overwrite {out}"
os.chdir(SANA)
from fake_quant.collect.collect import collect
import importlib

module = importlib.import_module("fake_quant.collect.collect")
prompts = json.loads((BASE / "calibration_prompts.json").read_text())


class ArchivedDataset:
    def __init__(self):
        self.prompts = [x["prompt"] for x in prompts]

    def __len__(self):
        return len(prompts)

    def __iter__(self):
        return iter([dict(x, mode="Gen") for x in prompts])


module.get_dataset = lambda args: ArchivedDataset()
config = json.loads((BASE / "plain.json").read_text())
args = argparse.Namespace(
    config=str(SANA / config["config"]),
    model_path=str(SANA / "checkpoints/SANA1.5_1.6B_1024px.pth"),
    save_path=str(out),
    seed=42,
    height=1024,
    width=1024,
    guidance_scale=4.5,
    pag_guidance_scale=1.0,
    num_inference_steps=20,
    prompt_file=str(BASE / "calibration_prompts.json"),
)
collect(args)
files = sorted((out / "cache").glob("*.pt"))
assert len(files) > 0
(out / "collection_complete.json").write_text(
    json.dumps(
        dict(
            prompts=32,
            forward_inputs=len(files),
            steps=20,
            seed=42,
            guidance=4.5,
            pag=1.0,
            files=[dict(file=f.name, sha256=digest(f)) for f in files],
        ),
        indent=2,
    )
)
