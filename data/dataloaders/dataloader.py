import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from preprocessing.add_colourspaces import add_channels

class GreenSpillDataset(Dataset):
    def __init__(self, spill_images, gt_images, transform_fn):
        self.spill_images = spill_images
        self.gt_images = gt_images
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.spill_images)

    def __getitem__(self, idx):
        # Input (spill)
        input_img_array = self.transform_fn(self.spill_images[idx])
        input_tensor = torch.tensor(input_img_array).permute(2, 0, 1)  # (5, H, W)

        # Target (gt)
        gt_rgb = cv2.cvtColor(self.gt_images[idx], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        target_tensor = torch.tensor(gt_rgb).permute(2, 0, 1)   # (3, H, W)

        return input_tensor, target_tensor

def get_dataloader(spill_images, gt_images, batch_size, num_workers=2):
    dataset = GreenSpillDataset(spill_images=spill_images, gt_images=gt_images, transform_fn=add_channels)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    num_samples_in_batch = len(dataloader.dataset)
    return dataloader, num_samples_in_batch

