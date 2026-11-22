# ruff: noqa: E402
"""Run the archived MJHQ metric functions and GenEval, with local model paths."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for key in ("HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE"):
    os.environ[key] = str(ROOT / "cache/huggingface/hub")
os.environ.update(
    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TORCH_HOME=str(ROOT / "cache/torch")
)
import torch

sys.path.insert(0, str(ROOT.parent / "third_party/mmdetection"))
os.environ["PYTHONPATH"] = (
    str(ROOT.parent / "third_party/mmdetection")
    + os.pathsep
    + os.environ.get("PYTHONPATH", "")
)

p = argparse.ArgumentParser()
p.add_argument("run", type=Path)
p.add_argument("--smoke", action="store_true")
a = p.parse_args()
run = a.run.resolve()
torch.manual_seed(42)
metrics_path = ROOT / "eval/MJHQ-30k/evaluation/metrics.py"
spec = importlib.util.spec_from_file_location("archived_mjhq_metrics", metrics_path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
gen = run / "mjhq30k/images"
items = m.load_generated_metadata(gen)
result = {}
if not a.smoke:
    assert len(items) == 1000, len(items)
    result["FID"] = m.calculate_fid(str(gen), str(ROOT / "eval/MJHQ-30k/data"))
    kid, std = m.calculate_kid(str(gen), str(ROOT / "eval/MJHQ-30k/data"))
    result.update(KID_mean=kid, KID_std=std)
    (run / "mjhq_metrics.json").write_text(json.dumps(result, indent=2))
payload = {
    "pairs": [
        {
            "prompt": x["prompt"],
            "image_path": str(gen / x["local_id"] / "samples" / f"{x['local_id']}.png"),
        }
        for x in items
    ]
}
payload_file = run / "reward_input.json"
payload_file.write_text(json.dumps(payload))
subprocess.run(
    [
        sys.executable,
        str(metrics_path.parent / "imagereward_runner.py"),
        "--input_json",
        str(payload_file),
        "--output_json",
        str(run / "imagereward.json"),
        "--ckpt",
        str(ROOT / "checkpoints/metrics/ImageReward/ImageReward.pt"),
        "--med_config",
        str(ROOT / "checkpoints/metrics/ImageReward/med_config.json"),
    ],
    check=True,
)
result["ImageReward"] = json.loads((run / "imagereward.json").read_text())["mean"]
base = ROOT / "eval/geneval"
subprocess.run(
    [
        sys.executable,
        str(base / "evaluation/evaluate_images.py"),
        str(run / "geneval/images"),
        "--outfile",
        str(run / "geneval/results.jsonl"),
        "--model-config",
        str(
            ROOT.parent
            / "third_party/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
        ),
        "--model-path",
        str(base / "model"),
        "--clip-checkpoint",
        str(ROOT / "checkpoints/metrics/ViT-L-14.pt"),
    ],
    check=True,
)
rows = [json.loads(x) for x in (run / "geneval/results.jsonl").read_text().splitlines()]
tags = sorted({r["tag"] for r in rows})
scores = {
    tag: sum(bool(r["correct"]) for r in rows if r["tag"] == tag)
    / sum(r["tag"] == tag for r in rows)
    for tag in tags
}
result.update(
    GenEval=sum(scores.values()) / len(scores), GenEval_tasks=scores, smoke_only=a.smoke
)
(run / "evaluation_results.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
