import os

import torch


class CalibSampleDataset(torch.utils.data.Dataset):
    def __init__(self, path, seed=42):
        del seed
        if not os.path.exists(path):
            raise FileNotFoundError(f"The specified path does not exist: {path}")
        files = sorted(f for f in os.listdir(path) if f.endswith(".pt"))
        self.files = [os.path.join(path, f) for f in files]

    def __getitem__(self, index):
        return torch.load(self.files[index], map_location="cpu")

    def __len__(self):
        return len(self.files)

    def build_loader(self, **kwargs) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self, batch_size=None, shuffle=False, **kwargs)
