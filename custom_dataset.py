import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
import os


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

        # 2. Augmentations (Applied AFTER cropping)
        if is_training:
            self.aug_pipeline = iaa.Sequential(
                [
                    # Geometric
                    iaa.Affine(rotate=(-180, 180)),
                    iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),
                    iaa.Fliplr(0.5),
                    iaa.Flipud(0.5),
                    # Color
                    iaa.GammaContrast((0.8, 1.2)),
                    iaa.AddToHueAndSaturation((-10, 10)),
                ]
            )
        else:
            self.aug_pipeline = None  # No augmentation for validation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        task_id = int(row["class_index"])

        # 1. Load Full Image (3000x3000px)
        # Optimization: We read full image.
        # For huge WSIs, libraries like OpenSlide are better,
        # but for 3000x3000px PNGs, plt.imread is fast enough (~30MB RAM).
        image = plt.imread(row["image_path"])
        mask = plt.imread(row["mask_path"])

        # Formatting
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[2] > 3:
            image = image[:, :, :3]
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Normalize to uint8 [0, 255] for processing
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

        # 2. INTELLIGENT CROPPING
        h, w = image.shape[:2]

        # Default: Random Crop
        top = np.random.randint(0, h - self.crop_h)
        left = np.random.randint(0, w - self.crop_w)

        # Force Foreground Logic:
        # If training AND we have a mask AND it's a sparse class (or just 50% of time for all)
        # Let's apply it to ALL classes to be safe, especially Tuft/Cap.
        if self.is_training:
            # Find all pixels that are part of the object
            # (mask > 0 assuming binary mask 0/255)
            y_indices, x_indices = np.where(mask > 127)

            # If the image actually contains the object (is not empty)
            if len(y_indices) > 0:
                # 50% chance to force the crop to be centered on an object
                if np.random.rand() < 0.5:
                    # Pick a random pixel belonging to the object
                    idx = np.random.randint(len(y_indices))
                    center_y, center_x = y_indices[idx], x_indices[idx]

                    # Calculate Top-Left coordinate to center this pixel
                    top = center_y - (self.crop_h // 2)
                    left = center_x - (self.crop_w // 2)

                    # Clip to image boundaries (don't go outside)
                    top = np.clip(top, 0, h - self.crop_h)
                    left = np.clip(left, 0, w - self.crop_w)

        # Apply Crop
        image = image[top : top + self.crop_h, left : left + self.crop_w, :]
        mask = mask[top : top + self.crop_h, left : left + self.crop_w]

        # Mask needs channel dim for imgaug
        mask = mask[:, :, np.newaxis]

        # 3. Apply Augmentations (Rotation/Color)
        if self.aug_pipeline:
            image, mask = self.aug_pipeline(image=image, segmentation_maps=mask)

        # 4. Final Tensor Formatting
        image = image.transpose((2, 0, 1)).astype(np.float32) / 255.0
        mask = mask[:, :, 0]
        mask = (mask > 127).astype(np.float32)

        return image, mask, task_id, 0  # scale_id=0
