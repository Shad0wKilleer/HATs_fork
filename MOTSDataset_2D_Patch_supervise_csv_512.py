import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.ndimage
import imgaug.augmenters as iaa
from torch.utils import data


class MOTSDataSet(data.Dataset):
    def __init__(
        self,
        supervise_root,
        list_path,
        max_iters=None,
        crop_size=(64, 192, 192),
        mean=(128, 128, 128),
        scale=True,
        mirror=True,
        ignore_label=255,
        edge_weight=1,
    ):
        self.supervise_root = supervise_root
        # list_path, max_iters, crop_size, mean, scale, mirror, ignore_label are kept for interface compatibility
        self.edge_weight = edge_weight

        self.image_mask_aug = iaa.Sequential(
            [
                iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
                iaa.Affine(rotate=(-180, 180)),
                iaa.Affine(shear=(-16, 16)),
                iaa.Fliplr(0.5),
                iaa.ScaleX((0.75, 1.5)),
                iaa.ScaleY((0.75, 1.5)),
            ]
        )

        self.image_aug_color = iaa.Sequential(
            [
                iaa.GammaContrast((0, 2.0)),
                iaa.Add((-0.1, 0.1), per_channel=0.5),
            ]
        )

        self.image_aug_noise = iaa.Sequential(
            [
                iaa.CoarseDropout((0.0, 0.05), size_percent=(0.00, 0.25)),
                iaa.GaussianBlur(sigma=(0, 1.0)),
                iaa.AdditiveGaussianNoise(scale=(0, 0.1)),
            ]
        )

        self.image_aug_resolution = iaa.AverageBlur(k=(2, 8))

        self.image_aug_256 = iaa.Sequential(
            [iaa.MultiplyHueAndSaturation((-10, 10), per_channel=0.5)]
        )

        self.crop512 = iaa.CropToFixedSize(width=512, height=512, position="uniform")
        self.pad512 = iaa.PadToFixedSize(width=512, height=512, position="uniform")

        self.df_supervise = pd.read_csv(self.supervise_root)
        self.df_supervise.sample(frac=1)

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
        image = image[:, :, :3]
        label = label[:, :, :3]

        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)

        if image.shape[1] == 1024:
            cnt = 0
            image_i, label_i = self.crop512(images=image, heatmaps=label)

            while label_i.sum() > 0.8 * 512 * 512 * 3 and cnt <= 50:
                image_i, label_i = self.crop512(images=image, heatmaps=label)
                cnt += 1

            image, label = image_i, label_i

        elif image.shape[1] == 256:
            image, label = self.pad512(images=image, heatmaps=label)

        seed = np.random.rand(4)

        if seed[0] > 0.5:
            image, label = self.image_mask_aug(images=image, heatmaps=label)

        if seed[1] > 0.5:
            image = self.image_aug_color(images=image)

        if seed[2] > 0.5:
            image = self.image_aug_noise(images=image)

        label[label >= 0.5] = 1.0
        label[label < 0.5] = 0.0

        image = image[0].transpose((2, 0, 1))  # Channel x H x W
        label = label[0, :, :, 0]

        image = image.astype(np.float32)
        label = label.astype(np.uint8)

        if self.edge_weight:
            weight = scipy.ndimage.binary_dilation(label == 1, iterations=2) & ~label
        else:
            weight = np.ones(label.shape, dtype=label.dtype)

        label = label.astype(np.float32)

        return (
            image.copy(),
            label.copy(),
            weight.copy(),
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
        # list_path, max_iters, crop_size, mean, scale, mirror, ignore_label are kept for interface compatibility
        self.edge_weight = edge_weight

        self.df = pd.read_csv(self.root)
        self.df.sample(frac=1)

        self.pad1024 = iaa.PadToFixedSize(width=1024, height=1024, position="center")

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

        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)

        if image.shape[1] == 256 or image.shape[1] == 512:
            image, label = self.pad1024(images=image, heatmaps=label)

        label[label >= 0.5] = 1.0
        label[label < 0.5] = 0.0

        image = image[0].transpose((2, 0, 1))  # Channel x H x W
        label = label[0, :, :, 0]

        image = image.astype(np.float32)
        label = label.astype(np.float32)
        weight = np.ones(label.shape, dtype=label.dtype)

        return (
            image.copy(),
            label.copy(),
            weight.copy(),
            name,
            layer_id,
            task_id,
            scale_id,
        )
