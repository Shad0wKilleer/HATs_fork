from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
import os


class RenalDataset(Dataset):
    def __init__(self, csv_files, is_training=True, crop_size=(512, 512)):
        """
        Args:
            csv_files: List of file paths to your CSVs (e.g. ['dataset_pt_train.csv', ...])
            is_training: If True, applies random geometric and color augmentations.
            crop_size: The size of the patch to extract from the WSI (default 512x512).
        """
        self.is_training = is_training

        # 1. Load and Concatenate all CSVs
        # Your CSV format: class_index, image_path, mask_path (No Header)
        dataframes = []
        for csv_path in csv_files:
            if os.path.exists(csv_path):
                df = pd.read_csv(
                    csv_path,
                    header=None,
                    names=["class_index", "image_path", "mask_path"],
                )
                dataframes.append(df)
            else:
                print(f"Warning: CSV not found: {csv_path}")

        if len(dataframes) == 0:
            raise RuntimeError("No valid CSV files provided!")

        self.data = pd.concat(dataframes, ignore_index=True)
        print(f"Dataset Loaded: {len(self.data)} images from {len(csv_files)} files.")

        # 2. Define Augmentations
        # We use 'CropToFixedSize' to handle the 3000x3000 -> 512x512 transition.
        if is_training:
            # Mild Geometric Augmentations
            self.aug_geo = iaa.Sequential(
                [
                    iaa.Affine(
                        rotate=(-180, 180)
                    ),  # Rotation is standard for histology
                    iaa.Affine(
                        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}
                    ),  # Slight shift to avoid center-bias
                    iaa.Fliplr(0.5),  # Mirror horizontal
                    iaa.Flipud(0.5),  # Mirror vertical
                ]
            )

            # Mild Color Augmentations (Safe for pathology)
            self.aug_color = iaa.Sequential(
                [
                    iaa.GammaContrast((0.8, 1.2)),  # Simulates exposure differences
                    iaa.AddToHueAndSaturation(
                        (-10, 10)
                    ),  # Simulates slight stain variation
                ]
            )

            # Training: Randomly crop a 512x512 patch from the large image
            self.cropper = iaa.CropToFixedSize(
                width=crop_size[0], height=crop_size[1], position="uniform"
            )
        else:
            # Validation: Deterministic center crop for consistency
            self.cropper = iaa.CropToFixedSize(
                width=crop_size[0], height=crop_size[1], position="center"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        # 1. Load Image & Mask
        img_path = row["image_path"]
        mask_path = row["mask_path"]

        # Matplotlib reads PNGs as floats (0-1) or ints (0-255). We normalize later.
        image = plt.imread(img_path)
        mask = plt.imread(mask_path)

        # Handle formatting (Remove Alpha channel if present)
        if image.ndim == 3 and image.shape[2] > 3:
            image = image[:, :, :3]
        if mask.ndim == 3:
            mask = mask[:, :, 0]  # Flatten mask to 2D (H, W)

        # Ensure Unsigned Int 8-bit [0, 255] for imgaug processing
        # (If image is float 0.0-1.0, convert it)
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        if mask.dtype != np.uint8:
            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)

        # Mask needs a channel dimension for imgaug: (H, W) -> (H, W, 1)
        mask = mask[:, :, np.newaxis]

        # 2. Apply Augmentations
        # First: Crop the massive 3000x3000px image to 512x512px
        image, mask = self.cropper(image=image, segmentation_maps=mask)

        if self.is_training:
            # Geometric (Rotations/Flips) -> Applies to Image AND Mask
            image, mask = self.aug_geo(image=image, segmentation_maps=mask)
            # Color (Stain/Light) -> Applies to Image ONLY
            image = self.aug_color(image=image)

        # 3. Final Tensor Formatting
        # Image: (H, W, C) -> (C, H, W), Normalized to [0.0, 1.0]
        image = image.transpose((2, 0, 1)).astype(np.float32) / 255.0

        # Mask: (H, W, 1) -> (H, W), Binary 0.0 or 1.0
        mask = mask[:, :, 0]
        mask = (mask > 127).astype(np.float32)  # Threshold at 0.5 (127/255)

        # Metadata
        task_id = int(row["class_index"])
        scale_id = 0  # Defaulting to 0 since all data is at the same magnification

        return image, mask, task_id, scale_id
