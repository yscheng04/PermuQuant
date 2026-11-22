# PermuQuant for Sana-1.5 1.6B

Run commands below from `Sana/`.

## Download models

Download [Sana-1.5 1.6B](https://huggingface.co/Efficient-Large-Model/SANA1.5_1.6B_1024px), the [Gemma text encoder](https://huggingface.co/Efficient-Large-Model/gemma-2-2b-it), and [DC-AE](https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers):

```bash
conda activate permuquant

hf download Efficient-Large-Model/SANA1.5_1.6B_1024px \
  checkpoints/SANA1.5_1.6B_1024px.pth --local-dir .
hf download Efficient-Large-Model/gemma-2-2b-it \
  --cache-dir cache/huggingface/hub
hf download mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers \
  --cache-dir cache/huggingface/hub
```

The directory layout should be:

```text
Sana/
├── checkpoints/
│   └── SANA1.5_1.6B_1024px.pth
└── cache/huggingface/hub/
    ├── models--Efficient-Large-Model--gemma-2-2b-it/
    └── models--mit-han-lab--dc-ae-f32c32-sana-1.1-diffusers/
```

## Calibration

Collect the recorded calibration prompts, then search permutations. Use `--variant plain` for PermuQuant, or `--variant hadamard` for PermuQuant-H, which applies a Hadamard rotation before permutation.

```bash
conda activate permuquant
CUDA_VISIBLE_DEVICES=0 python fake_quant/cli/collect.py --output fake_quant/cli/runs/forward-inputs

CUDA_VISIBLE_DEVICES=0 python fake_quant/cli/calibrate.py --variant plain \
  --output fake_quant/cli/runs/plain-calibration
CUDA_VISIBLE_DEVICES=0 python fake_quant/cli/calibrate.py --variant hadamard \
  --output fake_quant/cli/runs/hadamard-calibration
```

## Generate images

Generate 200 GenEval and 1,000 MJHQ images for each variant:

```bash
CUDA_VISIBLE_DEVICES=0 python fake_quant/cli/run.py --variant plain \
  --checkpoint fake_quant/cli/runs/plain-calibration/reorder.pt \
  --output fake_quant/cli/runs/plain
CUDA_VISIBLE_DEVICES=0 python fake_quant/cli/run.py --variant hadamard \
  --checkpoint fake_quant/cli/runs/hadamard-calibration/reorder.pt \
  --output fake_quant/cli/runs/hadamard
```

## Evaluation

Prepare the [evaluation assets](../README.md#evaluation-assets), then use the shared evaluation environment:

```bash
conda activate permuquant-eval
CUDA_VISIBLE_DEVICES=0 python fake_quant/cli/evaluate.py fake_quant/cli/runs/plain
CUDA_VISIBLE_DEVICES=0 python fake_quant/cli/evaluate.py fake_quant/cli/runs/hadamard
```

Results are saved to `evaluation_results.json` in each output directory: FID, KID, ImageReward and GenEval. Display KID multiplied by 1,000.
