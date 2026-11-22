import torch
import os
import random

class CalibSampleDataset(torch.utils.data.Dataset):
    def __init__(self, path, seed = 42):
        if os.path.exists(path):
            self.path = path
        else:
            raise FileNotFoundError(f"The specified path does not exist: {path}")
        self.files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pt')]
        random.Random(seed).shuffle(self.files)

    def __getitem__(self, index):
        file_path = self.files[index]
        sample = torch.load(file_path)
        # sample['input_kwargs']['update_past_key_values'] = False
        return sample

    def __len__(self):
        return len(self.files)
    
    def build_loader(self, **kwargs) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self, batch_size=None, shuffle=False, **kwargs)