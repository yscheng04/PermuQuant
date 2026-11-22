import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import random
import json
from fake_quant.utils import hash_str_to_int

class CalibDataset(Dataset):
    def __init__(self, 
                 prompt_folder="data/coco-caption/coco_karpathy_test.json",
                 num_samples=4,   
                 image_transform=None): 
        """
        最简易的校准数据集。
        如果提供了 image_folder，就从文件夹读图；
        如果没有提供，就生成纯色图片（用于测试流程）。
        """
        self.num_samples = num_samples

        self.image_transform = image_transform
        self.prompts = []

        if prompt_folder and os.path.exists(prompt_folder):
            with open(prompt_folder, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # data 是一个列表，每个元素是一个字典
                for item in data:
                    # item['caption'] 是一个列表，我们取第一个作为 prompt
                    if 'caption' in item and len(item['caption']) > 0:
                        self.prompts.append(item['caption'][0])
        else:
            raise ValueError(f"Prompt file path'{prompt_folder}' is incorrect.")
        num_prompts = self.num_samples
        assert len(self.prompts) >= num_prompts, "Prompt file does not contain enough prompts."
        self.prompts = random.sample(self.prompts, num_prompts)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        file_name = f"{idx:03d}.png"

        return {       
            "prompt": prompt,
            "file_name": file_name,
        }

# 测试一下
if __name__ == "__main__":
    ds = CalibDataset(num_samples=5)
    print(f"样本 0 图片: {ds[0]['image'].size}")