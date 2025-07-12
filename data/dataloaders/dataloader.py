import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from preprocessing.add_colourspaces import add_channels

class GreenSpillDataset(Dataset):
    def __init__(self, spill_images, gt_images, use_extra_channels=False):
        self.spill_images = spill_images
        self.gt_images = gt_images
        self.use_extra_channels = use_extra_channels

    def __len__(self):
        return len(self.spill_images)

    def __getitem__(self, idx):
        # Input (spill)
        if self.use_extra_channels:
            # r, g, b, h, greenness
            input_img = add_channels(bgr_img=self.spill_images[idx])
        else:
            input_img = cv2.cvtColor(
                self.spill_images[idx],
                cv2.COLOR_BGR2RGB
            ).astype(np.float32) / 255.0

        input_tensor = torch.tensor(input_img).permute(2, 0, 1)  # (C, H, W)

        # Target (gt)
        gt_rgb = cv2.cvtColor(self.gt_images[idx], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        target_tensor = torch.tensor(gt_rgb).permute(2, 0, 1)   # (3, H, W)

        return input_tensor, target_tensor

def get_dataloader(spill_images, gt_images, batch_size, num_workers=2, use_extra_channels=False):
    """
    Creates batches of images for dataloader
    :param spill_images: List of spill images
    :param gt_images: List of ground truth images
    :param batch_size: Batch size
    :param num_workers: Number of subprocesses to use
    :param use_extra_channels: Whether to add extra channels or not
    :return: Dataloader and number of batches
    """
    dataset = GreenSpillDataset(
        spill_images=spill_images,
        gt_images=gt_images,
        use_extra_channels=use_extra_channels
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    num_batches = len(dataloader)
    return dataloader, num_batches

