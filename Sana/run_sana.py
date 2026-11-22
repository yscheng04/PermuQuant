import torch
from app.sana_pipeline import SanaPipeline
from torchvision.utils import save_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator = torch.Generator(device=device).manual_seed(42)

sana = SanaPipeline("configs/sana1-5_config/1024ms/Sana_1600M_1024px_allqknorm_bf16_lr2e5.yaml")
sana.from_pretrained("hf://Efficient-Large-Model/SANA1.5_1.6B_1024px/checkpoints/SANA1.5_1.6B_1024px.pth")

image = sana(
    prompt='一只赛博朋克风格的猫，旁边有霓虹灯写着 "Sana"',
    height=1024,
    width=1024,
    guidance_scale=4.5,
    pag_guidance_scale=1.0,
    num_inference_steps=20,
    generator=generator,
)

save_image(image, "output/sana15_native.png", nrow=1, normalize=True, value_range=(-1, 1))