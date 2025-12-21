import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.ndimage
import torch
from torch.utils import data
from torchvision.transforms import v2
from torchvision import tv_tensors


class MOTSDataSet(data.Dataset):
    def __init__(
        self,
        supervise_root,
        list_path,
        max_iters=None,
        crop_size=(64, 192, 192),  # Kept for signature compatibility, but unused
        mean=(128, 128, 128),  # Kept for signature compatibility
        scale=True,
        mirror=True,
        ignore_label=255,
        edge_weight=1,
    ):
        self.supervise_root = supervise_root
        self.edge_weight = edge_weight

        # --- Geometric Augmentations (Applied to Image AND Mask) ---
        # Logic matches:
        # iaa.Affine(translate, rotate, shear) -> v2.RandomAffine
        # iaa.Fliplr -> v2.RandomHorizontalFlip
        # iaa.ScaleX/Y -> v2.RandomAffine(scale=...) (Isotropic approx)
        # iaa.CropToFixedSize -> v2.RandomCrop
        self.geo_transforms = v2.Compose(
            [
                v2.RandomAffine(
                    degrees=180,
                    translate=(0.2, 0.2),
                    shear=(-16, 16, -16, 16),
                    scale=(0.75, 1.5),
                    interpolation=v2.InterpolationMode.BILINEAR,
                ),
                v2.RandomHorizontalFlip(p=0.5),
                v2.Pad(
                    padding=512, padding_mode="reflect"
                ),  # Pad before crop to ensure size is sufficient
                v2.RandomCrop(size=(512, 512)),
            ]
        )

        # --- Color/Noise Augmentations (Image Only) ---
        # Logic matches:
        # iaa.GammaContrast -> v2.RandomGamma
        # iaa.Add -> v2.ColorJitter(brightness)
        # iaa.CoarseDropout -> v2.RandomErasing
        # iaa.GaussianBlur -> v2.GaussianBlur
        # iaa.AdditiveGaussianNoise -> v2.GaussianNoise
        # iaa.MultiplyHueAndSaturation -> v2.ColorJitter(hue, saturation)
        self.color_transforms = v2.Compose(
            [
                v2.RandomApply([v2.RandomGamma(log_gamma=(0.5, 2.0))], p=0.5),
                v2.RandomApply(
                    [v2.ColorJitter(brightness=0.1)], p=0.5
                ),  # Matches Add(-0.1, 0.1)
                v2.RandomApply(
                    [v2.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 1.0))], p=0.5
                ),
                v2.RandomApply([v2.GaussianNoise(mean=0.0, sigma=0.1)], p=0.5),
                v2.RandomApply(
                    [v2.RandomErasing(scale=(0.0, 0.05), ratio=(0.3, 3.3), value=0)],
                    p=0.5,
                ),
                v2.RandomApply([v2.ColorJitter(hue=0.05, saturation=0.05)], p=0.5),
            ]
        )

        self.df_supervise = pd.read_csv(self.supervise_root)
        self.now_len = len(self.df_supervise)
        print("{} images are loaded!".format(self.now_len))

    def __len__(self):
        return self.now_len

    def __getitem__(self, index):
        datafiles = self.df_supervise.iloc[index]

        image = plt.imread(datafiles["image_path"])
        label = plt.imread(datafiles["label_path"])

        name = datafiles["name"]
        layer_id = datafiles["layer_id"]
        task_id = datafiles["task_id"]
        scale_id = datafiles["scale_id"]

        # Ensure 3 channels
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        else:
            image = image[:, :, :3]

        if label.ndim == 2:
            label = np.stack([label] * 3, axis=-1)
        else:
            label = label[:, :, :3]

        # Convert to Torch Tensors
        # Permute from (H, W, C) to (C, H, W)
        img_t = torch.from_numpy(image).permute(2, 0, 1).float()
        lbl_t = torch.from_numpy(label).permute(2, 0, 1).float()

        # Wrap mask in TVTensor so V2 knows to treat it as a mask (Nearest Neighbor interpolation)
        lbl_t = tv_tensors.Mask(lbl_t)
        img_t = tv_tensors.Image(img_t)

        # Apply Geometric Transforms (Jointly)
        img_t, lbl_t = self.geo_transforms(img_t, lbl_t)

        # Apply Color Transforms (Image Only)
        img_t = self.color_transforms(img_t)

        # Post-processing to match original output
        # Binarize label: Original logic was label[label >= 0.5] = 1.0
        lbl_t = (lbl_t >= 0.5).float()

        # Select single channel for label: Original logic returned (H, W)
        lbl_out = lbl_t[0, :, :]
        img_out = img_t  # (C, H, W)

        # Convert back to numpy for edge_weight calculation (using scipy)
        # Note: We could implement edge_weight in torch, but using scipy preserves exact logic
        lbl_np = lbl_out.numpy()

        if self.edge_weight:
            weight = scipy.ndimage.binary_dilation(lbl_np == 1, iterations=2) & ~(
                lbl_np == 1
            )
            weight = weight.astype(np.float32)
        else:
            weight = np.ones(lbl_np.shape, dtype=np.float32)

        return (
            img_out,  # Tensor (3, H, W)
            lbl_out,  # Tensor (H, W)
            torch.from_numpy(weight),  # Tensor (H, W)
            name,
            layer_id,
            task_id,
            scale_id,
        )


class MOTSValDataSet(data.Dataset):
    def __init__(
        self,
        root,
        list_path,
        max_iters=None,
        crop_size=(256, 256),
        mean=(128, 128, 128),
        scale=False,
        mirror=False,
        ignore_label=255,
        edge_weight=1,
    ):
        self.root = root
        self.edge_weight = edge_weight
        self.df = pd.read_csv(self.root)

        # Validation transform: Just padding to 1024 (center)
        # Original: iaa.PadToFixedSize(width=1024, height=1024, position="center")
        self.transforms = v2.Compose(
            [
                v2.Pad(padding=1024, padding_mode="constant", fill=0),  # Pad liberally
                v2.CenterCrop(size=(1024, 1024)),  # Crop to desired size
            ]
        )

        print("{} images are loaded!".format(len(self.df)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        datafiles = self.df.iloc[index]
        image = plt.imread(datafiles["image_path"])
        label = plt.imread(datafiles["label_path"])

        name = datafiles["name"]
        layer_id = datafiles["layer_id"]
        task_id = datafiles["task_id"]
        scale_id = datafiles["scale_id"]

        image = image[:, :, :3]
        label = label[:, :, :3]

        # Convert to Torch Tensors
        img_t = torch.from_numpy(image).permute(2, 0, 1).float()
        lbl_t = torch.from_numpy(label).permute(2, 0, 1).float()

        # Apply Transforms
        if img_t.shape[1] < 1024 or img_t.shape[2] < 1024:
            # Calculate padding needed to center
            # V2 CenterCrop handles "larger than crop" but for smaller, we need Pad.
            # Easier approach: Pad to target size or larger, then CenterCrop.
            pad_h = max(0, 1024 - img_t.shape[1])
            pad_w = max(0, 1024 - img_t.shape[2])
            padding = [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2]

            # Apply manual padding to ensure centering
            img_t = v2.functional.pad(img_t, padding, fill=0)
            lbl_t = v2.functional.pad(lbl_t, padding, fill=0)

        # Ensure 1024x1024
        cropper = v2.CenterCrop(size=(1024, 1024))
        img_t = cropper(img_t)
        lbl_t = cropper(lbl_t)

        # Post-process
        lbl_t = (lbl_t >= 0.5).float()
        lbl_out = lbl_t[0, :, :]

        weight = torch.ones_like(lbl_out)

        return (
            img_t,
            lbl_out,
            weight,
            name,
            layer_id,
            task_id,
            scale_id,
        )
