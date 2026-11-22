import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import sys
import tempfile
import subprocess
from typing import Dict, List


def load_generated_metadata(generated_dir):
    items = []
    for subdir in sorted(Path(generated_dir).iterdir()):
        if not subdir.is_dir():
            continue
        metadata_path = subdir / "metadata.jsonl"
        if not metadata_path.exists():
            continue
        with open(metadata_path, "r") as f:
            line = f.readline().strip()
            if not line:
                continue
            data = json.loads(line)
        data["local_id"] = subdir.name
        items.append(data)
    return items


def load_mjhq_metadata(reference_dir):
    with open(Path(reference_dir) / "meta_data.json", "r") as f:
        return json.load(f)


def load_image_tensor(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return tensor


def calculate_fid(generated_dir, reference_dir, device="cuda", batch_size=32):
    from torchmetrics.image.fid import FrechetInceptionDistance

    gen_meta = load_generated_metadata(generated_dir)
    mjhq_meta = load_mjhq_metadata(reference_dir)

    gen_paths = []
    ref_paths = []

    for meta in gen_meta:
        local_id = meta["local_id"]
        image_id = meta["image_id"]
        category = mjhq_meta[image_id]["category"]

        gen_paths.append(Path(generated_dir) / local_id / "samples" / f"{local_id}.png")
        ref_paths.append(Path(reference_dir) / category / f"{image_id}.jpg")

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    for i in tqdm(range(0, len(gen_paths), batch_size), desc="FID gen"):
        batch = torch.stack([load_image_tensor(p) for p in gen_paths[i:i+batch_size]]).to(device)
        fid.update(batch, real=False)

    for i in tqdm(range(0, len(ref_paths), batch_size), desc="FID ref"):
        batch = torch.stack([load_image_tensor(p) for p in ref_paths[i:i+batch_size]]).to(device)
        fid.update(batch, real=True)

    return fid.compute().item()


def calculate_kid(generated_dir, reference_dir, device="cuda", batch_size=32):
    from torchmetrics.image.kid import KernelInceptionDistance

    gen_meta = load_generated_metadata(generated_dir)
    mjhq_meta = load_mjhq_metadata(reference_dir)

    gen_paths = []
    ref_paths = []

    for meta in gen_meta:
        local_id = meta["local_id"]
        image_id = meta["image_id"]
        category = mjhq_meta[image_id]["category"]

        gen_paths.append(Path(generated_dir) / local_id / "samples" / f"{local_id}.png")
        ref_paths.append(Path(reference_dir) / category / f"{image_id}.jpg")

    kid = KernelInceptionDistance(feature=2048, normalize=True).to(device)

    for i in tqdm(range(0, len(gen_paths), batch_size), desc="KID gen"):
        batch = torch.stack([load_image_tensor(p) for p in gen_paths[i:i+batch_size]]).to(device)
        kid.update(batch, real=False)

    for i in tqdm(range(0, len(ref_paths), batch_size), desc="KID ref"):
        batch = torch.stack([load_image_tensor(p) for p in ref_paths[i:i+batch_size]]).to(device)
        kid.update(batch, real=True)

    kid_mean, kid_std = kid.compute()
    return kid_mean.item(), kid_std.item()


def calculate_clipscore(generated_dir, device="cuda", batch_size=32):
    import clip

    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()

    gen_meta = load_generated_metadata(generated_dir)
    pairs = []

    for meta in gen_meta:
        local_id = meta["local_id"]
        prompt = meta["prompt"]
        img_path = Path(generated_dir) / local_id / "samples" / f"{local_id}.png"
        pairs.append((img_path, prompt))

    scores = []
    with torch.no_grad():
        for i in tqdm(range(0, len(pairs), batch_size), desc="CLIPScore"):
            batch = pairs[i:i+batch_size]
            images = torch.stack([preprocess(Image.open(p).convert("RGB")) for p, _ in batch]).to(device)
            texts = clip.tokenize([t for _, t in batch], truncate=True).to(device)

            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            sim = (image_features * text_features).sum(dim=-1)
            scores.extend((2.5 * torch.clamp(sim, min=0)).cpu().tolist())

    return float(np.mean(scores))


def calculate_hpsv2(generated_dir, device="cuda", max_samples=100):
    import hpsv2

    gen_meta = load_generated_metadata(generated_dir)

    # Randomly sample if more than max_samples
    if len(gen_meta) > max_samples:
        indices = np.random.choice(len(gen_meta), max_samples, replace=False)
        gen_meta = [gen_meta[i] for i in indices]

    image_paths = []
    prompts = []
    for meta in gen_meta:
        local_id = meta["local_id"]
        image_paths.append(str(Path(generated_dir) / local_id / "samples" / f"{local_id}.png"))
        prompts.append(meta["prompt"])

    # Note: hpsv2.score() expects a single prompt string, not a list
    # Score each image with its corresponding prompt
    scores = []
    for img_path, prompt in tqdm(zip(image_paths, prompts), desc="HPSv2"):
        score = hpsv2.score([img_path], prompt)
        scores.append(score[0])
    return float(np.mean(scores))

def calculate_imagereward(generated_dir, device="cuda", max_samples=None):
    import os
    import sys
    import json
    import tempfile
    import subprocess

    gen_meta = load_generated_metadata(generated_dir)

    if max_samples is not None and len(gen_meta) > max_samples:
        indices = np.random.choice(len(gen_meta), max_samples, replace=False)
        gen_meta = [gen_meta[i] for i in indices]

    pairs = []
    for meta in gen_meta:
        local_id = meta["local_id"]
        img_path = Path(generated_dir) / local_id / "samples" / f"{local_id}.png"
        pairs.append({
            "image_path": str(img_path),
            "prompt": meta["prompt"],
        })

    if len(pairs) == 0:
        raise ValueError("No valid image/prompt pairs found for ImageReward evaluation.")

    payload = {"pairs": pairs}

    imagereward_python = "/home/yongsencheng/.conda/envs/imagereward/bin/python"
    imagereward_runner = str(Path(__file__).resolve().parent / "imagereward_runner.py")
    imagereward_ckpt = "/home/yongsencheng/hdd/models/ImageReward/ImageReward.pt"
    imagereward_med_config = "/home/yongsencheng/hdd/models/ImageReward/med_config.json"

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f_in:
        json.dump(payload, f_in)
        input_json = f_in.name

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f_out:
        output_json = f_out.name

    try:
        proc = subprocess.run(
            [
                imagereward_python,
                imagereward_runner,
                "--input_json", input_json,
                "--output_json", output_json,
                "--ckpt", imagereward_ckpt,
                "--med_config", imagereward_med_config,
                "--device", device,
            ],
            capture_output=True,
            text=True,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                "ImageReward subprocess failed.\n"
                f"STDOUT:\n{proc.stdout}\n\n"
                f"STDERR:\n{proc.stderr}"
            )

        with open(output_json, "r") as f:
            result = json.load(f)

        return float(result["mean"])

    finally:
        if os.path.exists(input_json):
            os.remove(input_json)
        if os.path.exists(output_json):
            os.remove(output_json)


def evaluate_mjhq30k(
    generated_dir: str,
    reference_dir: str,
    output_file: str,
    calculate_fid_score: bool = True,
    calculate_kid_score: bool = True,
    calculate_clip_score: bool = True,
    calculate_hps: bool = True,
    calculate_imagereward_score: bool = False,
    device: str = "cuda",
    batch_size: int = 32,
) -> Dict[str, float]:
    """Run full MJHQ-30K evaluation.

    Args:
        generated_dir: Directory containing generated images (with subdirs like 00000/samples/00000.png)
        reference_dir: Directory containing reference images (MJHQ-30K data dir with meta_data.json and category subdirs)
        output_file: Path to save evaluation results
        calculate_fid_score: Whether to calculate FID
        calculate_kid_score: Whether to calculate KID
        calculate_clip_score: Whether to calculate CLIPScore
        calculate_hps: Whether to calculate HPS v2
        device: Device to use for computation
        batch_size: Batch size for CLIPScore computation

    Returns:
        Dictionary containing evaluation results
    """
    results = {}

    print("\n" + "=" * 50)
    print("[MJHQ-30K] Starting Evaluation...")
    print("=" * 50)

    # FID
    if calculate_fid_score:
        try:
            fid_score = calculate_fid(generated_dir, reference_dir, device=device, batch_size=batch_size)
            results['fid'] = fid_score
        except Exception as e:
            print(f"[FID] Error: {e}")
            import traceback
            traceback.print_exc()
            results['fid'] = None

    # KID
    if calculate_kid_score:
        try:
            kid_mean, kid_std = calculate_kid(generated_dir, reference_dir, device=device, batch_size=batch_size)
            results['kid_mean'] = kid_mean
            results['kid_std'] = kid_std
        except Exception as e:
            print(f"[KID] Error: {e}")
            import traceback
            traceback.print_exc()
            results['kid'] = None

    # CLIPScore
    if calculate_clip_score:
        try:
            clip_score = calculate_clipscore(generated_dir, device=device, batch_size=batch_size)
            results['clipscore'] = clip_score
        except Exception as e:
            print(f"[CLIPScore] Error: {e}")
            import traceback
            traceback.print_exc()
            results['clipscore'] = None

    # HPS v2
    if calculate_hps:
        try:
            hps_score = calculate_hpsv2(generated_dir, device=device)
            results['hpsv2'] = hps_score
        except Exception as e:
            print(f"[HPSv2] Error: {e}")
            import traceback
            traceback.print_exc()
            results['hpsv2'] = None

    # ImageReward
    if calculate_imagereward_score:
        try:
            imagereward_score = calculate_imagereward(generated_dir, device=device)
            results["imagereward"] = imagereward_score
        except Exception as e:
            print(f"[ImageReward] Error: {e}")
            import traceback
            traceback.print_exc()
            results["imagereward"] = None

    # Save results
    print("\n" + "=" * 50)
    print("[MJHQ-30K] Evaluation Results:")
    for metric, value in results.items():
        if value is not None:
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: Failed")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results
