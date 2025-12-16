import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss4MOTS(nn.Module):
    """
    Computes Dice Loss for Multi-Object Tracking and Segmentation (MOTS) style data.
    Input: Logits [B, C, H, W]
    Target: One-Hot [B, C, H, W]
    """

    def __init__(self, num_classes=2, smooth=1e-5):
        super(DiceLoss4MOTS, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets, weights=None):
        # 1. Apply Softmax to get probabilities (Network outputs Logits)
        probs = F.softmax(logits, dim=1)

        # 2. Flatten for calculation (Batch * Class * Pixels)
        # We flatten spatial dimensions (H, W) but keep Batch/Class separation first
        # Then we sum over spatial dims.

        # probs: [B, C, H, W]
        # targets: [B, C, H, W]

        intersection = torch.sum(probs * targets, dim=(2, 3))
        union = torch.sum(probs, dim=(2, 3)) + torch.sum(targets, dim=(2, 3))

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # dice_score is now [B, C]

        # 3. Apply Weights (if any)
        if weights is not None:
            if isinstance(weights, list):
                weights = torch.tensor(weights).to(logits.device)
            # weights: [C] -> expand to [1, C]
            weights = weights.view(1, -1)
            dice_score = dice_score * weights

        # 4. Average over Class and Batch
        dice_loss = 1.0 - dice_score

        return dice_loss.mean()


class CELoss4MOTS(nn.Module):
    """
    Cross Entropy Loss wrapper that accepts One-Hot targets.
    Input: Logits [B, C, H, W]
    Target: One-Hot [B, C, H, W]
    """

    def __init__(self, num_classes=2, ignore_index=255, weight=None):
        super(CELoss4MOTS, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # We convert the list weight to a tensor if provided
        self.register_buffer("weight", torch.tensor(weight) if weight else None)

    def forward(self, logits, targets):
        # Standard CrossEntropyLoss expects:
        # Input: [B, C, H, W] (Logits)
        # Target: [B, H, W] (Indices)

        # Since our targets are One-Hot [B, C, H, W], we convert them to indices
        # argmax(1) turns [B, 2, H, W] -> [B, H, W] with values 0 or 1
        target_indices = torch.argmax(targets, dim=1)

        # Apply standard CE
        return F.cross_entropy(
            logits, target_indices, weight=self.weight, ignore_index=self.ignore_index
        )
