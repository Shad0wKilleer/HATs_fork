import argparse
import os, sys

import pandas as pd

# sys.path.append("/Data4/HATs/EfficientSAM_token_dynamichead_logits") # This path may need adjustment
import glob
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os.path as osp

# --- PyTorch Automatic Mixed Precision (AMP) ---
from torch.cuda.amp import autocast, GradScaler

from MOTSDataset_2D_Patch_supervise_csv_512 import MOTSDataSet as MOTSDataSet
from MOTSDataset_2D_Patch_supervise_csv_512 import MOTSValDataSet as MOTSValDataSet

import random
import timeit
import loss_functions.loss_2D as loss

from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import DataLoader

start = timeit.default_timer()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- PrPSeg Model Import ---
from unet2D_Dodnet_scale_token import UNet2D as UNet2D_scale


def HATs_learning(
    images,
    labels,
    batch_size,
    scales,
    model,
    now_task,
    weight,
    loss_seg_DICE,
    loss_seg_CE,
    term_seg_Dice,
    term_seg_BCE,
    term_all,
    HATs_matrix,
    semi_ratio,
    area_ratio,
):
    # This is the "Anatomy Loss" function from the PrPSeg paper
    for ii in range(len(HATs_matrix[1])):
        now_task_semi = ii
        if now_task_semi == now_task:
            continue
        now_relative = HATs_matrix[now_task][now_task_semi]
        now_area_ratio = area_ratio[now_task][now_task_semi]

        if now_relative == 0:
            continue

        semi_preds = model(
            images, torch.ones(batch_size).cuda() * now_task_semi, scales
        )

        "Only use dice in semi-supervised learning"
        if now_relative == 1:  # Superset
            semi_labels = 1 - labels
            semi_labels = loss.one_hot_3D(semi_labels.long())
            semi_seg_Dice, semi_seg_BCE, semi_all = loss.get_loss(
                images, semi_preds, semi_labels, weight, loss_seg_DICE, loss_seg_CE
            )
            term_seg_Dice -= semi_ratio * semi_seg_Dice * now_area_ratio
            term_all -= semi_ratio * semi_seg_Dice * now_area_ratio

        elif now_relative == -1:  # Subset
            semi_labels = labels
            semi_preds = semi_labels.unsqueeze(1).repeat(1, 2, 1, 1) * semi_preds
            semi_labels = loss.one_hot_3D(semi_labels.long())
            semi_seg_Dice, semi_seg_BCE, semi_all = loss.get_loss(
                images, semi_preds, semi_labels, weight, loss_seg_DICE, loss_seg_CE
            )
            term_seg_Dice += semi_ratio * semi_seg_Dice * now_area_ratio
            term_all += semi_ratio * semi_seg_Dice * now_area_ratio

        elif now_relative == 2:  # Mutually Exclusive
            semi_labels = labels
            semi_labels = loss.one_hot_3D(semi_labels.long())
            semi_seg_Dice, semi_seg_BCE, semi_all = loss.get_loss(
                images, semi_preds, semi_labels, weight, loss_seg_DICE, loss_seg_CE
            )
            term_seg_Dice -= semi_ratio * semi_seg_Dice * now_area_ratio
            term_all -= semi_ratio * semi_seg_Dice * now_area_ratio

    return term_seg_Dice, term_seg_BCE, term_all


def main():
    # --- Command-Line Arguments ---
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/Data4/HATs/data/train/data_list.csv",
        help="path to data list csv",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default="/Data4/HATs/data/val/data_list.csv",
        help="path to val data list csv",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="batch_size")
    parser.add_argument("--val_batch_size", type=int, default=8, help="batch_size")
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers")
    parser.add_argument("--val_num_workers", type=int, default=4, help="num_workers")
    parser.add_argument(
        "--input_size",
        type=str,
        default="512,512",
        help="width and height of input images",
    )
    parser.add_argument("--num_epochs", type=int, default=200, help="num_epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="learning_rate"
    )
    parser.add_argument(
        "--snapshot_dir",
        type=str,
        default="/Data4/HATs/model/HATs_all",
        help="path to snapshot",
    )
    parser.add_argument("--task_num", type=int, default=15, help="task numbers")
    parser.add_argument(
        "--semi_ratio", type=float, default=0.5, help="semi_supervised ratio"
    )
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    args = parser.parse_args()

    output_folder = args.snapshot_dir
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cudnn.enabled = True
    cudnn.benchmark = True
    torch.manual_seed(2022)
    torch.cuda.manual_seed(2022)
    np.random.seed(2022)
    random.seed(2022)

    # --- Build PrPSeg Model ---
    print(f"Building PrPSeg (UNet2D_scale) model for {args.task_num} classes...")
    model = UNet2D_scale(num_classes=args.task_num, num_scale=4, weight_std=False)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    torch.cuda.set_device(args.local_rank)
    model.cuda()

    # --- Initialize PyTorch AMP GradScaler ---
    scaler = GradScaler()

    model = nn.DataParallel(model)

    # --- Data Loaders ---
    print("Loading training data from:", args.data_path)
    trainloader = DataLoader(
        MOTSDataSet(args.data_path, crop_size=(args.input_size, args.input_size)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("Loading validation data from:", args.val_data_path)
    valloader = DataLoader(
        MOTSValDataSet(
            args.val_data_path, crop_size=(args.input_size, args.input_size)
        ),
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.val_num_workers,
        pin_memory=True,
    )

    # --- Universal Proposition Matrix (HATs_matrix) ---
    # Set args.task_num = 6
    print("Defining 6x6 Universal Proposition Matrix (HATs_matrix)...")
    HATs_matrix = np.zeros((args.task_num, args.task_num))

    # --- MAPPING ---
    # Your class 0 (dt)   = original class 5
    # Your class 1 (pt)   = original class 6
    # Your class 2 (cap)  = original class 7
    # Your class 3 (tuft) = original class 8
    # Your class 4 (art)  = original class 9
    # Your class 5 (ptc)  = original class 10

    # All these functional units are mutually exclusive (2) from each other...
    HATs_matrix[0, 1] = 2  # dt vs pt
    HATs_matrix[0, 2] = 2  # dt vs cap
    HATs_matrix[0, 3] = 2  # dt vs tuft
    HATs_matrix[0, 4] = 2  # dt vs art
    HATs_matrix[0, 5] = 2  # dt vs ptc

    HATs_matrix[1, 0] = 2  # pt vs dt
    HATs_matrix[1, 2] = 2  # pt vs cap
    HATs_matrix[1, 3] = 2  # pt vs tuft
    HATs_matrix[1, 4] = 2  # pt vs art
    HATs_matrix[1, 5] = 2  # pt vs ptc

    HATs_matrix[2, 0] = 2  # cap vs dt
    HATs_matrix[2, 1] = 2  # cap vs pt
    # ... except for the cap/tuft relationship
    HATs_matrix[2, 3] = 1  # cap (2) COVERS tuft (3)
    HATs_matrix[2, 4] = 2  # cap vs art
    HATs_matrix[2, 5] = 2  # cap vs ptc

    HATs_matrix[3, 0] = 2  # tuft vs dt
    HATs_matrix[3, 1] = 2  # tuft vs pt
    HATs_matrix[3, 2] = -1  # tuft (3) IS COVERED BY cap (2)
    HATs_matrix[3, 4] = 2  # tuft vs art
    HATs_matrix[3, 5] = 2  # tuft vs ptc

    HATs_matrix[4, 0] = 2  # art vs dt
    HATs_matrix[4, 1] = 2  # art vs pt
    HATs_matrix[4, 2] = 2  # art vs cap
    HATs_matrix[4, 3] = 2  # art vs tuft
    HATs_matrix[4, 5] = 2  # art vs ptc

    HATs_matrix[5, 0] = 2  # ptc vs dt
    HATs_matrix[5, 1] = 2  # ptc vs pt
    HATs_matrix[5, 2] = 2  # ptc vs cap
    HATs_matrix[5, 3] = 2  # ptc vs tuft
    HATs_matrix[5, 4] = 2  # ptc vs art

    # --- Area Ratio Matrix (for loss weighting) ---
    # You MUST also update the Area array to match your 6 classes
    Area = np.zeros((args.task_num))
    Area[0] = 0.097  # original Area[5] (dt)
    Area[1] = 0.360  # original Area[6] (pt)
    Area[2] = 0.619  # original Area[7] (cap)
    Area[3] = 0.466  # original Area[8] (tuft)
    Area[4] = 0.083  # original Area[9] (art)
    Area[5] = 0.002  # original Area[10] (ptc)

    Area_ratio = np.zeros((args.task_num, args.task_num))
    for xi in range(0, args.task_num):
        for yi in range(0, args.task_num):
            if Area[xi] > Area[yi]:
                Area_ratio[xi, yi] = Area[yi] / Area[xi]
            else:
                Area_ratio[xi, yi] = Area[xi] / Area[yi]

    # --- Loss Functions ---
    loss_seg_DICE = loss.DiceLoss().cuda()
    loss_seg_CE = loss.CELoss().cuda()

    best_dice = 0.0

    # --- Training Loop ---
    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in range(args.num_epochs):
        # --- Training Phase ---
        model.train()
        train_F1 = np.zeros(args.task_num)
        train_Dice = np.zeros(args.task_num)
        train_TPR = np.zeros(args.task_num)
        train_PPV = np.zeros(args.task_num)

        for i_iter, batch in enumerate(trainloader):
            images, labels, labels_weight, name, now_task, scales = batch
            images = images.cuda()
            labels = labels.cuda()
            labels_weight = labels_weight.cuda()
            now_task = now_task.cuda()
            scales = scales.cuda()

            optimizer.zero_grad()

            # --- AMP: Run forward pass in autocast context ---
            with autocast():
                preds = model(images, now_task, scales)

                # --- Standard Supervised Loss ---
                term_seg_Dice, term_seg_BCE, term_all = loss.get_loss(
                    images, preds, labels, labels_weight, loss_seg_DICE, loss_seg_CE
                )

                # --- Anatomy Loss (Semi-supervised) ---
                term_seg_Dice_semi, term_seg_BCE_semi, term_all_semi = HATs_learning(
                    images,
                    labels,
                    args.batch_size,
                    scales,
                    model,
                    now_task,
                    labels_weight,
                    loss_seg_DICE,
                    loss_seg_CE,
                    term_seg_Dice,
                    term_seg_BCE,
                    term_all,
                    HATs_matrix,
                    args.semi_ratio,
                    Area_ratio,
                )

                term_all = term_all_semi  # Use the combined loss

            # --- AMP: Scale loss and perform backward pass ---
            scaler.scale(term_all).backward()
            scaler.step(optimizer)
            scaler.update()

            # --- Calculate Training Metrics ---
            preds_numpy = preds.data.cpu().numpy()
            labels_numpy = labels.data.cpu().numpy()
            preds_numpy = np.argmax(preds_numpy, axis=1)

            for bi in range(args.batch_size):
                now_task_cpu = int(now_task[bi].data.cpu().numpy())
                score = f1_score(
                    labels_numpy[bi].flatten(),
                    preds_numpy[bi].flatten(),
                    average=None,
                    zero_division=0,
                )
                train_F1[now_task_cpu] += score[1]
                # ... (other metric calculations) ...

        print(
            f"Epoch {epoch}/{args.num_epochs} Training Stats: F1={np.mean(train_F1)}, Dice={np.mean(train_Dice)}"
        )

        # --- Validation Phase ---
        if (epoch % 1 == 0 and epoch >= 0) or (epoch >= args.num_epochs - 1):
            model.eval()
            val_F1 = np.zeros(args.task_num)
            val_Dice = np.zeros(args.task_num)
            val_TPR = np.zeros(args.task_num)
            val_PPV = np.zeros(args.task_num)

            df = pd.DataFrame(columns=["class", "F1", "Dice", "TPR", "PPV"])

            print("Validating...")
            for i_iter, batch in enumerate(valloader):
                images, labels, name, now_task, scales = batch
                images = images.cuda()
                labels = labels.cuda()
                now_task = now_task.cuda()
                scales = scales.cuda()

                with torch.no_grad():
                    # --- AMP: Run validation in autocast context ---
                    with autocast():
                        preds = model(images, now_task, scales)

                # --- Calculate Validation Metrics ---
                preds_numpy = preds.data.cpu().numpy()
                labels_numpy = labels.data.cpu().numpy()
                preds_numpy = np.argmax(preds_numpy, axis=1)

                for bi in range(args.val_batch_size):
                    if torch.sum(labels[bi]) == 0:
                        continue

                    now_task_cpu = int(now_task[bi].data.cpu().numpy())
                    score = f1_score(
                        labels_numpy[bi].flatten(),
                        preds_numpy[bi].flatten(),
                        average=None,
                        zero_division=0,
                    )
                    val_F1[now_task_cpu] += score[1]
                    # ... (other metric calculations) ...

            # --- Save Validation Results and Model Checkpoint ---
            avg_val_Dice = val_Dice / (len(valloader))
            print(f"Epoch {epoch} Validation Dice: {np.mean(avg_val_Dice)}")

            if np.mean(avg_val_Dice) > best_dice:
                best_dice = np.mean(avg_val_Dice)

                # --- Save metrics to CSV ---
                # This block now saves metrics for your 6 classes
                df.loc[0] = [
                    "0_dt",
                    val_F1[0].item(),
                    val_Dice[0].item(),
                    val_TPR[0].item(),
                    val_PPV[0].item(),
                ]
                df.loc[1] = [
                    "1_pt",
                    val_F1[1].item(),
                    val_Dice[1].item(),
                    val_TPR[1].item(),
                    val_PPV[1].item(),
                ]
                df.loc[2] = [
                    "2_cap",
                    val_F1[2].item(),
                    val_Dice[2].item(),
                    val_TPR[2].item(),
                    val_PPV[2].item(),
                ]
                df.loc[3] = [
                    "3_tuft",
                    val_F1[3].item(),
                    val_Dice[3].item(),
                    val_TPR[3].item(),
                    val_PPV[3].item(),
                ]
                df.loc[4] = [
                    "4_art",
                    val_F1[4].item(),
                    val_Dice[4].item(),
                    val_TPR[4].item(),
                    val_PPV[4].item(),
                ]
                df.loc[5] = [
                    "5_ptc",
                    val_F1[5].item(),
                    val_Dice[5].item(),
                    val_TPR[5].item(),
                    val_PPV[5].item(),
                ]

                df.to_csv(os.path.join(output_folder, "validation_result.csv"))

                # --- Save Model ---
                print(f"New best model with Dice: {best_dice}. Saving...")
                model_save_path = osp.join(
                    args.snapshot_dir, f"PrPSeg_best_model_epoch_{epoch}.pth"
                )

                # --- AMP: Save scaler state with model and optimizer ---
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),  # Save the scaler state
                }
                torch.save(checkpoint, model_save_path)


if __name__ == "__main__":
    main()
