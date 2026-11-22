"""Model-local helpers retained from the validated generation path."""

import random
import torch
from PIL import Image
from app.sana_pipeline import SanaPipeline


def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    sample = image_tensor.detach().cpu().float()
    sample = sample.mul(0.5).add(0.5).clamp(0, 1)
    ndarr = (
        sample.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to(torch.uint8)
        .numpy()
    )
    return Image.fromarray(ndarr)


def load_sana_components(args):
    sana = SanaPipeline(args.config)
    sana.from_pretrained(args.model_path)
    return {"pipeline": sana, "device": get_device()}


@torch.inference_mode()
def generate_image(prompt, components, args, seed, prompt_index=-1):
    seed_everything(seed)
    sana = components["pipeline"]
    generator = torch.Generator(device=sana.device).manual_seed(seed)
    image = sana(
        prompt=prompt,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        pag_guidance_scale=args.pag_guidance_scale,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
    )
    return tensor_to_pil(image[0])
