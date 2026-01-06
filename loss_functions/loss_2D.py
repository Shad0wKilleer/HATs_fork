import torch
import torch.nn.functional as F
import torch.nn as nn


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction="mean"):
        super().__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target, weight):
        assert predict.shape[0] == target.shape[0], (
            "predict & target batch size don't match"
        )
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        weight = weight.contiguous().view(weight.shape[0], -1)

        num = torch.sum(torch.mul(predict, target) * weight, dim=1)
        den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + self.smooth

        dice_score = 2 * num / den
        dice_loss = 1 - dice_score

        # Avoid potential division by zero if all targets are -1
        valid_mask = target[:, 0] != -1
        if valid_mask.sum() > 0:
            # I am still dividing by valid_mask.shape[0] (batch_size) instead
            # of valid_mask.sum() (which will give number of valid images in
            # this batch). Let suppose we have 2 valid images and batch size is
            # 4. Thus when we divide the loss of those two images with 4, we are
            # basically halving the loss for those 4 images. This is because:
            #   1. If the 2 images are outlier, they have less affect on loss
            #   2. All batches are equal. Though batches with more invalid images
            #      have less affect on loss but training is stable.
            dice_loss_avg = dice_loss[valid_mask].sum() / valid_mask.shape[0]
        else:
            dice_loss_avg = (
                dice_loss.sum() * 0
            )  # Return 0 with gradient connection if needed

        return dice_loss_avg


class DiceLoss4MOTS(nn.Module):
    # "weight" are not weights for edges but class weights to give a small class
    # like pt more loss than larger classes like capsules. This helps to keep
    # the losses for all classes on same class and prevent larger classes like
    # capsules/tufts to dominate smaller classes like pt/dt.

    # "num_classes" is the number of final outputs that the dynamic head outputs,
    # foreground and background. Thus num_classes=2 in our case. Do not
    # mix classes (foreground, background) with the tasks (dt, pt, capsules,
    # tufts etc)
    def __init__(self, weight=None, ignore_index=None, num_classes=3, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.dice = BinaryDiceLoss(**self.kwargs)

    # "weight" are the weights for edges so boundaries have more weights and
    # we can force the model to learn the boundaries more efficiently.
    def forward(self, predict, target, weight):
        total_loss = []
        predict = torch.sigmoid(predict)

        for i in range(self.num_classes):
            if i != self.ignore_index:
                dice_loss = self.dice(predict[:, i], target[:, i], weight)
                if self.weight is not None:
                    assert self.weight.shape[0] == self.num_classes, (
                        "Expect weight shape [{}], get[{}]".format(
                            self.num_classes, self.weight.shape[0]
                        )
                    )
                    dice_loss *= self.weight[i]
                total_loss.append(dice_loss)

        total_loss = torch.stack(total_loss)
        # Filter out NaNs if any exist. In python NaN != NaN
        total_loss = total_loss[total_loss == total_loss]

        return total_loss.sum() / total_loss.shape[0]


class CELoss4MOTS(nn.Module):
    def __init__(self, ignore_index=None, num_classes=3, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, predict, target, weight):
        assert predict.shape == target.shape, "predict & target shape do not match"

        total_loss = []
        for i in range(self.num_classes):
            if i != self.ignore_index:
                ce_loss = self.criterion(predict[:, i], target[:, i]) * weight
                ce_loss = torch.mean(ce_loss, dim=[1, 2])

                valid_mask = target[:, i, 0, 0] != -1
                if valid_mask.sum() > 0:
                    ce_loss_avg = ce_loss[valid_mask].sum() / valid_mask.shape[0]
                else:
                    ce_loss_avg = ce_loss.sum() * 0

                total_loss.append(ce_loss_avg)

        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss == total_loss]

        return total_loss.sum() / total_loss.shape[0]
