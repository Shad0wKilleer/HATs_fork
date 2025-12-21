import torch
import torch.nn.functional as F
import torch.nn as nn


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction="mean"):
        super(BinaryDiceLoss, self).__init__()
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
            dice_loss_avg = dice_loss[valid_mask].sum() / valid_mask.shape[0]
        else:
            dice_loss_avg = (
                dice_loss.sum() * 0
            )  # Return 0 with gradient connection if needed

        return dice_loss_avg


class DiceLoss4MOTS(nn.Module):
    def __init__(self, weight=None, ignore_index=None, num_classes=3, **kwargs):
        super(DiceLoss4MOTS, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.dice = BinaryDiceLoss(**self.kwargs)

    def forward(self, predict, target, weight):
        total_loss = []
        # UPDATED: F.sigmoid is deprecated
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
        # Filter out NaNs if any exist
        total_loss = total_loss[total_loss == total_loss]

        return total_loss.sum() / total_loss.shape[0]


class CELoss4MOTS(nn.Module):
    def __init__(self, ignore_index=None, num_classes=3, **kwargs):
        super(CELoss4MOTS, self).__init__()
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
