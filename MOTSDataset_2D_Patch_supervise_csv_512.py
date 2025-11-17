import numpy as np
import pandas as pd
import torch
from torch.utils import data
import scipy.ndimage
from PIL import Image

# Import the modern torchvision transforms
import torchvision.transforms.v2 as v2
from torchvision.tv_tensors import Image as TVImage, Mask as TVMask


class MOTSDataSet(data.Dataset):
    def __init__(
        self,
        supervise_root,
        max_iters=None,
        crop_size=(64, 192, 192),
        mean=(128, 128, 128),
        scale=True,
        mirror=True,
        ignore_label=255,
        edge_weight=1,
    ):
        self.supervise_root = supervise_root
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.edge_weight = edge_weight

        # --- Replaced imgaug with torchvision.v2 transforms ---

        # This pipeline applies identical transforms to both image and mask
        self.image_mask_aug = v2.Compose(
            [
                v2.RandomAffine(
                    degrees=180,
                    translate=(0.2, 0.2),
                    scale=(0.75, 1.5),  # Isotropic scaling
                    shear=(-16, 16),
                ),
                v2.RandomHorizontalFlip(p=0.5),
            ]
        )

        # This pipeline applies only to the image
        self.image_aug_color = v2.ColorJitter(brightness=0.2, contrast=(0.5, 1.8))

        # This pipeline applies noise and blur only to the image
        self.image_aug_noise = v2.Compose(
            [
                # Replaces CoarseDropout
                v2.RandomErasing(p=0.5, scale=(0.0, 0.05), ratio=(0.3, 3.3)),
                # Replaces GaussianBlur
                v2.GaussianBlur(kernel_size=3, sigma=(0.01, 1.0)),
                # Replaces AdditiveGaussianNoise
                v2.Lambda(
                    lambda x: x + torch.randn_like(x) * (0.1 * torch.rand(1)).item()
                ),
            ]
        )

        # This is the 512x512 random crop transform
        self.crop512 = v2.RandomCrop(512)

        # --- End of transform definitions ---

        self.df_supervise = pd.read_csv(self.supervise_root)
        # Fix: Re-assign the shuffled DataFrame
        self.df_supervise = self.df_supervise.sample(frac=1)

        self.now_len = len(self.df_supervise)
        print("{} training images are loaded!".format(self.now_len))

    def __len__(self):
        return self.now_len

    def __getitem__(self, index):
        datafiles = self.df_supervise.iloc[index]

        # Load images using PIL
        image = Image.open(datafiles["image_path"]).convert("RGB")
        label = Image.open(datafiles["label_path"]).convert("RGB")  # Load as RGB

        # Convert to Tensors first (range [0, 1])
        image = v2.functional.to_image(image)
        image = v2.functional.to_dtype(image, dtype=torch.float32, scale=True)

        label = v2.functional.to_image(label)
        label = v2.functional.to_dtype(label, dtype=torch.float32, scale=True)

        # Take only the first channel of the label and add a channel dim
        # Original logic was label = label[0,:,:,0]
        label = label[0, :, :].unsqueeze(0)

        # Wrap in tv_tensors to auto-apply transforms to image/mask
        image = TVImage(image)
        label = TVMask(label)

        # --- Handle Image Sizing ---
        if image.shape[1] == 1024:
            cnt = 0
            image_i, label_i = self.crop512(image, label)

            # Original logic to avoid empty crops
            while label_i.sum() > 0.8 * 512 * 512 and cnt <= 50:
                image_i, label_i = self.crop512(image, label)
                cnt += 1
            image, label = image_i, label_i

        elif image.shape[1] == 256:
            # Re-implement random padding (PadToFixedSize with position='uniform')
            pad_h = 512 - image.shape[1]
            pad_w = 512 - image.shape[2]

            # Get random padding amounts
            top = torch.randint(0, pad_h + 1, (1,)).item()
            left = torch.randint(0, pad_w + 1, (1,)).item()

            # Calculate padding [left, top, right, bottom]
            padding = [left, top, pad_w - left, pad_h - top]

            image = v2.functional.pad(image, padding, fill=0)
            label = v2.functional.pad(label, padding, fill=0)

        # --- Apply Augmentations ---
        if torch.rand(1) > 0.5:
            image, label = self.image_mask_aug(image, label)

        if torch.rand(1) > 0.5:
            image = self.image_aug_color(image)

        if torch.rand(1) > 0.5:
            image = self.image_aug_noise(image)

        # Binarize label (original logic)
        label = (label >= 0.5).to(torch.uint8)

        # Squeeze label from (1, H, W) to (H, W) for scipy
        label_squeezed = label.squeeze(0)

        # --- Calculate Edge Weight (original logic) ---
        if self.edge_weight:
            # Convert to numpy for scipy
            label_numpy = label_squeezed.numpy()
            weight_numpy = (
                scipy.ndimage.binary_dilation(label_numpy == 1, iterations=2)
                & ~label_numpy
            )
            weight = torch.from_numpy(weight_numpy.astype(np.float32))
        else:
            weight = torch.ones_like(label_squeezed, dtype=torch.float32)

        # Convert label to float for the model
        label = label_squeezed.to(torch.float32)

        return (
            image.to(torch.float32),
            label.to(torch.float32),
            weight,
            datafiles["name"],
            datafiles["layer_id"],
            datafiles["task_id"],
            datafiles["scale_id"],
        )


class MOTSValDataSet(data.Dataset):
    def __init__(
        self,
        root,
        list_path=None,  # list_path was unused, kept for compatibility
        max_iters=None,
        crop_size=(256, 256),
        mean=(128, 128, 128),
        scale=False,
        mirror=False,
        ignore_label=255,
        edge_weight=1,
    ):
        self.root = root
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.edge_weight = edge_weight

        self.df = pd.read_csv(self.root)
        # Fix: Re-assign the shuffled DataFrame
        self.df = self.df.sample(frac=1)

        print("{} validation images are loaded!".format(len(self.df)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        datafiles = self.df.iloc[index]

        # Load images using PIL
        image = Image.open(datafiles["image_path"]).convert("RGB")
        label = Image.open(datafiles["label_path"]).convert("RGB")  # Load as RGB

        # Convert to Tensors (range [0, 1])
        image = v2.functional.to_tensor(image)
        label = v2.functional.to_tensor(label)

        # Take only the first channel of the label
        label = label[0, :, :].unsqueeze(0)

        # --- Handle Image Sizing (Center Pad to 1024) ---
        if image.shape[1] == 256 or image.shape[1] == 512:
            pad_h = 1024 - image.shape[1]
            pad_w = 1024 - image.shape[2]

            # Calculate centered padding
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left

            padding = [left, top, right, bottom]

            image = v2.functional.pad(image, padding, fill=0)
            label = v2.functional.pad(label, padding, fill=0)

        # Binarize label
        label = (label >= 0.5).to(torch.float32)

        # Squeeze label from (1, H, W) to (H, W)
        label = label.squeeze(0)

        # Return a simple weight mask of ones (original logic)
        weight = torch.ones_like(label, dtype=torch.float32)

        return (
            image.to(torch.float32),
            label.to(torch.float32),
            weight,
            datafiles["name"],
            datafiles["layer_id"],
            datafiles["task_id"],
            datafiles["scale_id"],
        )
