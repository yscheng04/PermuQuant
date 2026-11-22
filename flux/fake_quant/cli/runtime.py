"""Model-local helpers retained from the validated generation path."""

import torch
from PIL import Image
from einops import rearrange
from flux.sampling import get_schedule, denoise, unpack
from flux.util import embed_watermark


def _resolve_module_device_ref(device_ref) -> torch.device | None:
    if isinstance(device_ref, int):
        return torch.device(f"cuda:{device_ref}")
    if isinstance(device_ref, str) and device_ref not in {"cpu", "disk"}:
        return torch.device(device_ref)
    return None


def get_module_device(module: torch.nn.Module, fallback: torch.device | None = None) -> torch.device:
    device_map = getattr(module, "hf_device_map", None)
    if isinstance(device_map, dict):
        for device_ref in device_map.values():
            resolved = _resolve_module_device_ref(device_ref)
            if resolved is not None:
                return resolved

    try:
        return next(module.parameters()).device
    except StopIteration:
        if fallback is None:
            raise ValueError(f"Unable to determine device for module {module.__class__.__name__}.")
        return fallback


def is_model_parallel(model: torch.nn.Module) -> bool:
    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict):
        return False

    devices = {
        resolved
        for device_ref in device_map.values()
        if (resolved := _resolve_module_device_ref(device_ref)) is not None
    }
    return len(devices) > 1


def get_parameter_device(module: torch.nn.Module, fallback: torch.device | None = None) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        if fallback is None:
            raise ValueError(f"Unable to determine parameter device for module {module.__class__.__name__}.")
        return fallback


def decode_with_ae(
    ae,
    x: torch.Tensor,
    fallback_device: torch.device,
) -> torch.Tensor:
    decode_device = get_parameter_device(ae.decoder, fallback=fallback_device)
    decoder_devices = sorted({str(param.device) for param in ae.decoder.parameters()})
    if len(decoder_devices) > 1:
        print(
            f"[Multi-GPU] AE decoder parameters span devices {decoder_devices}. Consolidating to {decode_device}."
        )

    ae.decoder.to(decode_device)
    decoder_devices = sorted({str(param.device) for param in ae.decoder.parameters()})
    if decoder_devices != [str(decode_device)]:
        raise RuntimeError(
            f"AE decoder device consolidation failed. Expected only {decode_device}, got {decoder_devices}."
        )

    if x.device != decode_device:
        x = x.to(decode_device)

    with torch.autocast(device_type=decode_device.type, dtype=torch.bfloat16):
        return ae.decode(x)


@torch.inference_mode()
def flux_generate(
    model,
    ae,
    inp: dict,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    device: torch.device,
    offload: bool = False,
) -> Image.Image:
    """Generate an image using FLUX model."""
    # Get timesteps
    timesteps = get_schedule(
        num_inference_steps,
        inp["img"].shape[1],
        shift=(model.params.guidance_embed),
    )

    # Denoise
    guidance = guidance_scale if model.params.guidance_embed else 0.0
    if not is_model_parallel(model):
        model = model.to(device)
        x = denoise(model, **inp, timesteps=timesteps, guidance=guidance)
        x = unpack(x.float(), height, width)

        if offload:
            model.cpu()
            torch.cuda.empty_cache()
            ae.decoder.to(x.device)

        x = decode_with_ae(ae, x, fallback_device=device)
    else:
        denoise_device = get_module_device(model, fallback=device)
        x = denoise(model, **inp, timesteps=timesteps, guidance=guidance)
        x = unpack(x.float(), height, width)
        x = decode_with_ae(ae, x, fallback_device=denoise_device)

    # Convert to PIL Image
    x = x.clamp(-1, 1)
    x = embed_watermark(x.float())
    x = rearrange(x[0], "c h w -> h w c")
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

    return img
