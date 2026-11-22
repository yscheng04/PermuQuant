# PermuQuant for FLUX.1-dev

Run commands below from `flux/`.

## Download models

First request access and accept the terms on [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev), then download the diffusion model, autoencoder and text encoders:

```bash
conda activate permuquant
hf auth login

hf download black-forest-labs/FLUX.1-dev flux1-dev.safetensors ae.safetensors \
  --local-dir checkpoints/black-forest-labs_FLUX.1-dev
hf download google/t5-v1_1-xxl \
  --exclude "*.h5" "*.msgpack" "*.onnx" \
  --local-dir checkpoints/t5-v1_1-xxl
hf download openai/clip-vit-large-patch14 \
  --exclude "*.h5" "*.msgpack" "*.onnx" "*.bin" \
  --local-dir checkpoints/clip-vit-large-patch14
```

The directory layout should be:

```text
flux/
└── checkpoints/
    ├── black-forest-labs_FLUX.1-dev/
    │   ├── flux1-dev.safetensors
    │   └── ae.safetensors
    ├── t5-v1_1-xxl/                
    └── clip-vit-large-patch14/     
```

## Calibration

Collect the recorded calibration prompts (two GPUs), then search permutations (one GPU). Use `--variant plain` for PermuQuant, or `--variant hadamard` for PermuQuant-H, which applies a Hadamard rotation before permutation.

```bash
conda activate permuquant
CUDA_VISIBLE_DEVICES=0,1 python fake_quant/cli/collect.py --output fake_quant/cli/runs/forward-inputs

CUDA_VISIBLE_DEVICES=0 python fake_quant/cli/calibrate.py --variant plain \
  --inputs fake_quant/cli/runs/forward-inputs/cache --output fake_quant/cli/runs/plain-calibration
CUDA_VISIBLE_DEVICES=0 python fake_quant/cli/calibrate.py --variant hadamard \
  --inputs fake_quant/cli/runs/forward-inputs/cache --output fake_quant/cli/runs/hadamard-calibration
```


## Generate images

Cache the evaluation text embeddings once, then generate 200 GenEval and 1,000 MJHQ images for each variant:

```bash
CUDA_VISIBLE_DEVICES=0 python fake_quant/cli/encode.py --output fake_quant/cli/runs/embeddings

CUDA_VISIBLE_DEVICES=0 python fake_quant/cli/run.py --variant plain \
  --checkpoint fake_quant/cli/runs/plain-calibration/reorder.pt \
  --embeddings fake_quant/cli/runs/embeddings --output fake_quant/cli/runs/plain
CUDA_VISIBLE_DEVICES=0 python fake_quant/cli/run.py --variant hadamard \
  --checkpoint fake_quant/cli/runs/hadamard-calibration/reorder.pt \
  --embeddings fake_quant/cli/runs/embeddings --output fake_quant/cli/runs/hadamard --offload-decode
```

## Evaluation

Prepare the [evaluation assets](../README.md#evaluation-assets), then use the shared evaluation environment:

```bash
conda activate permuquant-eval
CUDA_VISIBLE_DEVICES=0 python fake_quant/cli/evaluate.py fake_quant/cli/runs/plain
CUDA_VISIBLE_DEVICES=0 python fake_quant/cli/evaluate.py fake_quant/cli/runs/hadamard
```

Results are saved to `evaluation_results.json` in each output directory: FID, KID, ImageReward and GenEval. Display KID multiplied by 1,000.
