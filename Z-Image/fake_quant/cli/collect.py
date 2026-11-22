"""Replay the 64 recorded calibration prompts through the original collector."""

import argparse
import json
from pathlib import Path
import importlib

collector = importlib.import_module("fake_quant.collect.collect")
p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
out = a.output.resolve()
if out.exists():
    raise SystemExit("Refusing to overwrite collection output")
prompts = json.loads((Path(__file__).parent / "calibration_prompts.json").read_text())


class Dataset:
    def __len__(self):
        return len(prompts)

    def __getitem__(self, i):
        return {"file_name": f"{i:05d}.png", "prompt": prompts[i]}


collector.get_dataset = lambda args: Dataset()
collector.collect(argparse.Namespace(save_path=str(out), num_samples=len(prompts)))
(out / "collection_complete.json").write_text(
    json.dumps(
        {
            "prompts": len(prompts),
            "forward_inputs": len(list((out / "cache").glob("*.pt"))),
        }
    )
)
