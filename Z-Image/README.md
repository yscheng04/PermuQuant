# PermuQuant for Z-Image-Turbo

Run commands below from `Z-Image/`.

## Download models

Download [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), including the text encoder, tokenizer and VAE:

```bash
conda activate permuquant
hf download Tongyi-MAI/Z-Image-Turbo --local-dir ckpts/Z-Image-Turbo
```

The directory layout should be:

```text
Z-Image/
└── ckpts/
    └── Z-Image-Turbo/
        ├── model_index.json
        ├── scheduler/
        ├── text_encoder/
        ├── tokenizer/
        ├── transformer/
        └── vae/
```

## Calibration

Collect the recorded calibration prompts, then search permutations (one GPU). Use `--variant plain` for PermuQuant, or `--variant hadamard` for PermuQuant-H, which applies a Hadamard rotation before permutation.

```bash
conda activate permuquant
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. python fake_quant/cli/collect.py \
  --output fake_quant/cli/runs/forward-inputs

CUDA_VISIBLE_DEVICES=0 python fake_quant/cli/calibrate.py --variant plain --ratio 1.0 \
  --forward-cache fake_quant/cli/runs/forward-inputs/cache --output fake_quant/cli/runs/plain-calibration
CUDA_VISIBLE_DEVICES=0 python fake_quant/cli/calibrate.py --variant hadamard --ratio 1.0 \
  --forward-cache fake_quant/cli/runs/forward-inputs/cache --output fake_quant/cli/runs/hadamard-calibration
```

## Generate images

Generate 200 GenEval and 1,000 MJHQ images for each variant:

```bash
CUDA_VISIBLE_DEVICES=0 python fake_quant/cli/run.py --variant plain \
  --reorder-cache fake_quant/cli/runs/plain-calibration/reorder.pt \
  --output fake_quant/cli/runs/plain
CUDA_VISIBLE_DEVICES=0 python fake_quant/cli/run.py --variant hadamard \
  --reorder-cache fake_quant/cli/runs/hadamard-calibration/reorder.pt \
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
