"""Shared local paths and original FLUX functions for Table 1 reproduction."""

import hashlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FLUX = ROOT / "flux"
BASE = FLUX / "fake_quant/cli"
for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
    os.environ[key] = str(FLUX / "cache/huggingface/hub")
os.environ.update(
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    TORCH_HOME=str(FLUX / "cache/torch"),
    FLUX_MODEL=str(FLUX / "checkpoints/black-forest-labs_FLUX.1-dev/flux1-dev.safetensors"),
    FLUX_AE=str(FLUX / "checkpoints/black-forest-labs_FLUX.1-dev/ae.safetensors"),
)
sys.path[:0] = [str(FLUX / "src"), str(FLUX)]
import numpy as np  # noqa: E402
import torch  # noqa: E402
from flux.sampling import get_noise, prepare  # noqa: E402
from flux.util import load_ae, load_clip, load_flow_model, load_t5  # noqa: E402
from runtime import flux_generate  # noqa: E402


def seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_integrity():
    hashes = read_json(BASE / "audit.json")["source_sha256"]
    changed = [name for name, digest in hashes.items() if sha256(FLUX / name) != digest]
    if changed:
        raise RuntimeError(f"Original FLUX sources changed: {changed}")
    return hashes


def quant_config(variant):
    return dict(read_json(BASE / "audit.json")["historical"][variant]["config"]["quant_config"])


def encoders(device):
    return (
        load_t5(
            device,
            max_length=512,
            version=str(FLUX / "checkpoints/t5-v1_1-xxl"),
            local_files_only=True,
        ),
        load_clip(device, version=str(FLUX / "checkpoints/clip-vit-large-patch14"), local_files_only=True),
    )


__all__ = [
    "BASE",
    "FLUX",
    "ROOT",
    "encoders",
    "flux_generate",
    "get_noise",
    "load_ae",
    "load_flow_model",
    "prepare",
    "quant_config",
    "read_json",
    "seed",
    "sha256",
    "source_integrity",
    "torch",
    "write_json",
]
