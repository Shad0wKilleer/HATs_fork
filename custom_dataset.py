import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Modern PyTorch Transforms
from torchvision.transforms import v2
from torchvision import tv_tensors


class RenalDataset(Dataset):
    def __init__(self, csv_files, is_training=True, crop_size=(512, 512)):
        self.is_training = is_training
        self.crop_h, self.crop_w = crop_size

        # 1. Load Data
        dataframes = []
        for csv_path in csv_files:
            if os.path.exists(csv_path):
                # Columns: class_index, image_path, mask_path
                df = pd.read_csv(
                    csv_path,
                    header=None,
                    names=["class_index", "image_path", "mask_path"],
                )
                dataframes.append(df)
            else:
                print(f"Warning: CSV not found: {csv_path}")

        if not dataframes:
            raise RuntimeError("No CSV files loaded!")

        self.data = pd.concat(dataframes, ignore_index=True)
        print(f"Dataset Loaded: {len(self.data)} images.")

        # 2. Define Pure PyTorch Augmentations
        if is_training:
            self.transforms = v2.Compose(
                [
                    # Geometric (Applies to Image + Mask)
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomVerticalFlip(p=0.5),
                    v2.RandomAffine(
                        degrees=180,
                        translate=(0.1, 0.1),  # +/- 10% shift
                        scale=None,  # NO resizing/zooming
                        shear=None,
                    ),
                    # Color (Applies to Image ONLY automatically)
                    # mild brightness (gamma proxy) and stain jitter
                    v2.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
                    ),
                    # Ensure float32 0-1 range
                    v2.ToDtype(torch.float32, scale=True),
                ]
            )
        else:
            # Validation: Just normalize
            self.transforms = v2.Compose(
                [
                    v2.ToDtype(torch.float32, scale=True),
                ]
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        task_id = int(row["class_index"])

        # 1. Load Full Image (Numpy)
        image = plt.imread(row["image_path"])
        mask = plt.imread(row["mask_path"])

        # Formatting (Numpy)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[2] > 3:
            image = image[:, :, :3]
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Ensure uint8 [0, 255]
        if image.dtype != np.uint8:
            image = (
                (image * 255).astype(np.uint8)
                if image.max() <= 1.0
                else image.astype(np.uint8)
            )
        if mask.dtype != np.uint8:
            mask = (
                (mask * 255).astype(np.uint8)
                if mask.max() <= 1.0
                else mask.astype(np.uint8)
            )

        # 2. INTELLIGENT CROPPING (Numpy)
        # We do this in Numpy before converting to Tensor for speed/ease
        h, w = image.shape[:2]

        # Default: Random Crop
        top = np.random.randint(0, h - self.crop_h) if h > self.crop_h else 0
        left = np.random.randint(0, w - self.crop_w) if w > self.crop_w else 0

        # Force Foreground Logic (Training only)
        if self.is_training:
            y_indices, x_indices = np.where(mask > 127)
            if len(y_indices) > 0 and np.random.rand() < 0.5:
                idx = np.random.randint(len(y_indices))
                center_y, center_x = y_indices[idx], x_indices[idx]
                top = np.clip(center_y - (self.crop_h // 2), 0, h - self.crop_h)
                left = np.clip(center_x - (self.crop_w // 2), 0, w - self.crop_w)

        # Apply Crop
        image = image[top : top + self.crop_h, left : left + self.crop_w, :]
        mask = mask[top : top + self.crop_h, left : left + self.crop_w]

        # 3. Convert to Torch Tensors & Wrap for v2
        # Image: [H, W, C] -> [C, H, W]
        img_tensor = torch.from_numpy(image).permute(2, 0, 1)
        # Mask: [H, W] -> [1, H, W] (Needs channel dim)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        # WRAPPER: This tells v2 "This is an Image" and "This is a Mask"
        # So it knows to use Bilinear interpolation for Image and Nearest Neighbor for Mask
        img_wrapped = tv_tensors.Image(img_tensor)
        mask_wrapped = tv_tensors.Mask(mask_tensor)

        # 4. Apply Transforms
        if self.transforms:
            img_wrapped, mask_wrapped = self.transforms(img_wrapped, mask_wrapped)

        # 5. Final Cleanup
        # Remove wrappers, ensure mask is binary 0.0/1.0 float
        # Mask comes out as [1, H, W], we remove channel dim to get [H, W] for Dice Loss
        final_img = img_wrapped
        final_mask = (mask_wrapped[0] > 0.5).float()  # Threshold back to binary

        return final_img, final_mask, task_id, 0  # scale_id=0
