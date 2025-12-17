import argparse
import os
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import timeit

# Imports
from unet2D_Dodnet_scale_token import UNet2D
from custom_dataset import RenalDataset
from dataset_utils import TaskBalancedBatchSampler
import loss


def get_args():
    parser = argparse.ArgumentParser(description="PrPSeg Memory Optimized")
    parser.add_argument("--data_root", type=str, default="./")
    parser.add_argument("--save_dir", type=str, default="snapshots/PrPSeg_Run")
    parser.add_argument("--num_epochs", type=int, default=100)
    # Reduce batch size slightly to be safe, or try 4 with this new fix
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--num_scale", type=int, default=1)
    return parser.parse_args()


# ... (Keep get_renal_matrix and get_area_ratios same as before) ...
def get_renal_matrix():
    matrix = np.zeros((2, 2))
    matrix[0, 1] = 2
    matrix[1, 0] = 2
    return matrix


def get_area_ratios():
    return torch.tensor([[1.0, 3.0], [0.33, 1.0]]).float()


def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(args.save_dir)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    # 1. Dataset
    train_csvs = [
        os.path.join(args.data_root, "dataset_pt_train.csv"),
        os.path.join(args.data_root, "dataset_dt_train.csv"),
    ]

    # Check if using overfit or real training
    if not all(os.path.exists(f) for f in train_csvs):
        print("Standard CSVs not found, checking for overfit CSVs...")
        # Fallback logic or error handling here

    dataset = RenalDataset(train_csvs, is_training=True, crop_size=(512, 512))
    sampler = TaskBalancedBatchSampler(
        dataset.data["class_index"].tolist(), args.batch_size
    )
    train_loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=True
    )

    # 2. Model
    model = UNet2D(num_classes=args.num_classes, num_scale=args.num_scale).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scaler = torch.amp.GradScaler("cuda")  # Modern FP16

    dice_loss_func = loss.DiceLoss4MOTS(num_classes=2).cuda()
    ce_loss_func = loss.CELoss4MOTS(
        weight=[1.0, 1.0], num_classes=2, ignore_index=255
    ).cuda()

    matrix = get_renal_matrix()
    area_ratios = get_area_ratios().cuda()

    print("Starting Training with Gradient Accumulation...")
    start_time = timeit.default_timer()
    global_step = 0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        steps = 0

        for batch in train_loader:
            imgs, masks, task_ids, scale_ids = batch
            imgs, masks = imgs.cuda(non_blocking=True), masks.cuda(non_blocking=True)
            task_ids, scale_ids = (
                task_ids.cuda(non_blocking=True),
                scale_ids.cuda(non_blocking=True),
            )

            current_task = task_ids[0].item()

            # MEMORY FIX 1: Zero gradients at the START of the step
            optimizer.zero_grad()

            # ---------------------------
            # PART A: Supervised Learning
            # ---------------------------
            with torch.amp.autocast("cuda"):
                preds = model(imgs, task_ids, scale_ids)
                targets_onehot = torch.zeros_like(preds)
                targets_onehot[:, 0] = 1 - masks
                targets_onehot[:, 1] = masks

                loss_sup = dice_loss_func(preds, targets_onehot) + ce_loss_func(
                    preds, targets_onehot
                )

            # MEMORY FIX 2: Backward IMMEDIATELY to free graph
            scaler.scale(loss_sup).backward()

            # ---------------------------
            # PART B: Proposition Learning
            # ---------------------------
            loss_prop_accum = 0.0  # Just for logging

            for other_task in range(args.num_classes):
                relation = matrix[current_task][other_task]
                if relation == 0 or other_task == current_task:
                    continue

                # We start a NEW forward pass context for each logic check
                with torch.amp.autocast("cuda"):
                    w_prop = 0.1 * area_ratios[current_task, other_task]

                    other_ids = torch.full_like(task_ids, other_task)
                    preds_other = model(imgs, other_ids, scale_ids)

                    # Calculate Logic Loss
                    current_prop_loss = torch.tensor(0.0).cuda()

                    if relation == 2:  # Exclusive
                        t_zeros = torch.zeros_like(targets_onehot)
                        t_zeros[:, 0] = 1
                        current_prop_loss = w_prop * dice_loss_func(
                            preds_other, t_zeros
                        )

                    elif relation == 1:  # Superset
                        prob_other = torch.softmax(preds_other, dim=1)[:, 1]
                        bg_current = targets_onehot[:, 0]
                        intersection = (prob_other * bg_current).sum()
                        union = prob_other.sum() + bg_current.sum() + 1e-5
                        current_prop_loss = w_prop * (2 * intersection / union)

                    elif relation == -1:  # Subset
                        current_prop_loss = w_prop * dice_loss_func(
                            preds_other, targets_onehot
                        )

                # MEMORY FIX 3: Backward IMMEDIATELY
                # This accumulates gradients into the model without keeping the graph
                if current_prop_loss > 0:
                    scaler.scale(current_prop_loss).backward()
                    loss_prop_accum += current_prop_loss.item()

            # MEMORY FIX 4: Update Weights once all gradients are accumulated
            scaler.step(optimizer)
            scaler.update()

            # Logging (Sum of the parts)
            total_loss_val = loss_sup.item() + loss_prop_accum
            epoch_loss += total_loss_val
            steps += 1
            global_step += 1

            if steps % 20 == 0:
                print(
                    f"[Ep {epoch}][Step {steps}] T:{current_task} Loss:{total_loss_val:.4f}"
                )
                writer.add_scalar("Loss/Total", total_loss_val, global_step)

        print(f"=== Epoch {epoch} Avg Loss: {epoch_loss / steps:.4f} ===")
        if (epoch + 1) % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir, f"model_e{epoch + 1}.pth"),
            )

    print(f"Done. {timeit.default_timer() - start_time:.2f}s")
    writer.close()


if __name__ == "__main__":
    main()
