import argparse
import os
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import timeit

# --- Custom Project Imports ---
from unet2D_Dodnet_scale_token import UNet2D
from custom_dataset import RenalDataset
from dataset_utils import TaskBalancedBatchSampler
import loss  # Uses the loss.py we created


def get_args():
    parser = argparse.ArgumentParser(description="PrPSeg Training (Fixed)")
    parser.add_argument(
        "--data_root", type=str, default="./", help="Directory containing CSVs"
    )
    parser.add_argument("--save_dir", type=str, default="snapshots/PrPSeg_Run")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    # Model: 4 Classes (0:PT, 1:DT, 2:Cap, 3:Tuft)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--num_scale", type=int, default=1)
    return parser.parse_args()


def get_renal_matrix():
    """
    Values: 2 (Exclusive), 1 (Superset), -1 (Subset)
    """
    matrix = np.zeros((4, 4))

    # Exclusives
    matrix[0, 1] = 2
    matrix[1, 0] = 2  # PT <> DT
    matrix[0, 2] = 2
    matrix[2, 0] = 2  # PT <> Cap
    matrix[0, 3] = 2
    matrix[3, 0] = 2  # PT <> Tuft
    matrix[1, 2] = 2
    matrix[2, 1] = 2  # DT <> Cap
    matrix[1, 3] = 2
    matrix[3, 1] = 2  # DT <> Tuft

    # Capsule (2) covers Tuft (3)
    matrix[2, 3] = 1
    matrix[3, 2] = -1

    return matrix


def get_area_ratios():
    """
    Calculates weights to balance loss between large and small regions.
    Weight = Area(Current) / Area(Other)
    If Current is huge (PT) and Other is small (Tuft), we penalize less?
    Actually, let's keep it simple: normalize so small classes get attention.
    """
    # Relative sizes: PT(1.0), DT(0.3), Cap(0.1), Tuft(0.08)
    areas = [1.0, 0.3, 0.1, 0.08]
    ratios = np.zeros((4, 4))

    for i in range(4):
        for j in range(4):
            # Ratio to scale the loss.
            # If we are looking at a small object (Tuft), errors are costly -> High Weight
            ratios[i, j] = areas[i] / areas[j]

    return torch.tensor(ratios).float()


def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(args.save_dir)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    # --- 1. Dataset ---
    train_csvs = [
        os.path.join(args.data_root, "dataset_pt_train.csv"),
        os.path.join(args.data_root, "dataset_dt_train.csv"),
        os.path.join(args.data_root, "dataset_capsule_train.csv"),
        os.path.join(args.data_root, "dataset_tuft_train.csv"),
    ]

    dataset = RenalDataset(train_csvs, is_training=True, crop_size=(512, 512))
    sampler = TaskBalancedBatchSampler(
        dataset.data["class_index"].tolist(), args.batch_size
    )

    train_loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=True
    )

    # --- 2. Model & Logic ---
    model = UNet2D(num_classes=args.num_classes, num_scale=args.num_scale).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scaler = GradScaler()

    dice_loss_func = loss.DiceLoss4MOTS(num_classes=2).cuda()
    ce_loss_func = loss.CELoss4MOTS(
        weight=[1.0, 1.0], num_classes=2, ignore_index=255
    ).cuda()

    matrix = get_renal_matrix()
    # Move ratios to GPU for fast lookup
    area_ratios = get_area_ratios().cuda()

    print("Starting Training...")
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

            with autocast():
                # --- Supervised Loss ---
                preds = model(imgs, task_ids, scale_ids)
                targets_onehot = torch.zeros_like(preds)
                targets_onehot[:, 0] = 1 - masks
                targets_onehot[:, 1] = masks

                loss_sup = dice_loss_func(preds, targets_onehot) + ce_loss_func(
                    preds, targets_onehot
                )

                # --- Proposition Loss ---
                loss_prop = 0.0

                for other_task in range(args.num_classes):
                    relation = matrix[current_task][other_task]
                    if relation == 0 or other_task == current_task:
                        continue

                    # Weighting: Base 0.1 * Area Ratio
                    w_prop = 0.1 * area_ratios[current_task, other_task]

                    # Predict Other
                    other_ids = torch.full_like(task_ids, other_task)
                    preds_other = model(imgs, other_ids, scale_ids)

                    if relation == 2:  # Exclusive
                        t_zeros = torch.zeros_like(targets_onehot)
                        t_zeros[:, 0] = 1
                        loss_prop += w_prop * dice_loss_func(preds_other, t_zeros)

                    elif relation == 1:  # Superset
                        # Simplified: Intersection overlap
                        prob_other = torch.softmax(preds_other, dim=1)[:, 1]
                        bg_current = targets_onehot[:, 0]
                        intersection = (prob_other * bg_current).sum()
                        union = prob_other.sum() + bg_current.sum() + 1e-5
                        loss_prop += w_prop * (2 * intersection / union)

                    elif relation == -1:  # Subset
                        loss_prop += w_prop * dice_loss_func(
                            preds_other, targets_onehot
                        )

                total_loss = loss_sup + loss_prop

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # --- Logging ---
            epoch_loss += total_loss.item()
            steps += 1
            global_step += 1

            if steps % 20 == 0:
                print(
                    f"[Ep {epoch}][Step {steps}] T:{current_task} Loss:{total_loss.item():.4f}"
                )
                writer.add_scalar("Loss/Batch_Total", total_loss.item(), global_step)
                writer.add_scalar("Loss/Batch_Sup", loss_sup.item(), global_step)
                writer.add_scalar(
                    "Loss/Batch_Prop",
                    loss_prop if isinstance(loss_prop, float) else loss_prop.item(),
                    global_step,
                )

        # End of Epoch
        print(f"=== Epoch {epoch} Avg Loss: {epoch_loss / steps:.4f} ===")
        writer.add_scalar("Loss/Epoch_Avg", epoch_loss / steps, epoch)

        if (epoch + 1) % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir, f"model_e{epoch + 1}.pth"),
            )

    print(
        f"Training Complete. Total time: {(timeit.default_timer() - start_time) / 3600:.2f} hrs"
    )
    writer.close()


if __name__ == "__main__":
    main()
