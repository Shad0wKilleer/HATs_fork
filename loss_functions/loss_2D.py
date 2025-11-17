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

        dice_loss_avg = (
            dice_loss[target[:, 0] != -1].sum() / dice_loss[target[:, 0] != -1].shape[0]
        )

        return dice_loss_avg


class DiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, num_classes=3, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.dice = BinaryDiceLoss(**self.kwargs)

    def forward(self, predict, target, weight):
        total_loss = []
        predict = F.sigmoid(predict)

        for i in range(self.num_classes):
            if i != self.ignore_index:
                dice_loss = self.dice(predict[:, i], target[:, i], weight)
                if self.weight is not None:
                    assert self.weight.shape[0] == self.num_classes, (
                        "Expect weight shape [{}], get[{}]".format(
                            self.num_classes, self.weight.shape[0]
                        )
                    )
                    dice_loss *= self.weights[i]
                total_loss.append(dice_loss)

        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss == total_loss]

        return total_loss.sum() / total_loss.shape[0]


class CELoss(nn.Module):
    def __init__(self, ignore_index=None, num_classes=3, **kwargs):
        super(CELoss, self).__init__()
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

                ce_loss_avg = (
                    ce_loss[target[:, i, 0] != -1].sum()
                    / ce_loss[target[:, i, 0] != -1].shape[0]
                )

                total_loss.append(ce_loss_avg)

        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss == total_loss]

        return total_loss.sum() / total_loss.shape[0]


def one_hot_3D(label, num_classes):
    # label shape: (N, 1, H, W)
    # output shape: (N, C, H, W)
    label_onehot = torch.zeros(
        label.shape[0], num_classes, label.shape[2], label.shape[3]
    ).cuda()
    label_onehot.scatter_(1, label.long(), 1)
    return label_onehot


def get_loss(images, preds, labels, labels_weight, loss_seg_DICE, loss_seg_CE):
    labels_onehot = one_hot_3D(labels.unsqueeze(1), 2)

    term_seg_Dice = loss_seg_DICE(preds, labels_onehot, labels_weight)
    term_seg_BCE = loss_seg_CE(preds, labels_onehot, labels_weight)

    term_all = term_seg_Dice + term_seg_BCE

    return term_seg_Dice, term_seg_BCE, term_all


# The BinaryPrecisionLoss and Precision4MOTS classes were here,
# but they are not used by the training script, so they have been removed.
