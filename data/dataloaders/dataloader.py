import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class GreenSpillDataset(Dataset):
    def __init__(self, spill_images, gt_images):
        self.spill_images = spill_images
        self.gt_images = gt_images

    def __len__(self):
        return len(self.spill_images)

    def __getitem__(self, idx):
        # Input (spill)
        input_img = cv2.cvtColor(self.spill_images[idx], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        input_tensor = torch.tensor(input_img).permute(2, 0, 1)  # (3, H, W)

        # Target (gt)
        gt_rgb = cv2.cvtColor(self.gt_images[idx], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        target_tensor = torch.tensor(gt_rgb).permute(2, 0, 1)   # (3, H, W)

        return input_tensor, target_tensor

def get_dataloader(spill_images, gt_images, batch_size, num_workers=2):
    dataset = GreenSpillDataset(spill_images=spill_images, gt_images=gt_images)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    num_batches = len(dataloader)
    return dataloader, num_batches

