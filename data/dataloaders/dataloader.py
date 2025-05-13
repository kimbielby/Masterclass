import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class GreenSpillDataset(Dataset):
    def __init__(self, image_paths, target_paths, transform_fn):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        input_tensor = self.transform_fn(self.image_paths[idx])
        target_image = cv2.imread(self.target_paths[idx])
        target_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        input_tensor = torch.tensor(input_tensor).permute(2, 0, 1)  # (5, H, W)
        target_tensor = torch.tensor(target_rgb).permute(2, 0, 1)   # (3, H, W)

        return input_tensor, target_tensor

def get_dataloader(input_paths, target_paths, transform_fn):
    dataset = GreenSpillDataset(image_paths=input_paths, target_paths=target_paths, transform_fn=transform_fn)
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)
    num_samples_in_batch = len(dataloader.dataset)
    return dataloader, num_samples_in_batch

