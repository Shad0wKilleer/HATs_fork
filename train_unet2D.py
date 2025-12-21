import argparse
import os
import pandas as pd
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import os.path as osp
import timeit
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast

# Local imports
from MOTSDataset_2D_Patch_supervise_csv_512 import MOTSDataSet
from MOTSDataset_2D_Patch_supervise_csv_512 import MOTSValDataSet
import loss_functions.loss_2D as loss
from engine import Engine
from util_a.image_pool import ImagePool
from unet2D_Dodnet_scale_token import UNet2D as UNet2D_scale


def one_hot_3D(targets, C=2):
    targets_extend = targets.clone()
    targets_extend.unsqueeze_(1)  # convert to Nx1xHxW
    one_hot = torch.cuda.FloatTensor(
        targets_extend.size(0), C, targets_extend.size(2), targets_extend.size(3)
    ).zero_()
    one_hot.scatter_(1, targets_extend, 1)
    return one_hot


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_arguments():
    parser = argparse.ArgumentParser(description="UNet2D_HATs")
    parser.add_argument(
        "--trainset_dir", type=str, default="/Data/HATs/Data/train/data_list.csv"
    )
    parser.add_argument(
        "--valset_dir", type=str, default="/Data/HATs/Data/val/data_list.csv"
    )
    parser.add_argument("--train_list", type=str, default="list/MOTS/MOTS_train.txt")
    parser.add_argument("--val_list", type=str, default="list/MOTS/xx.txt")
    parser.add_argument("--edge_weight", type=float, default=1.0)
    parser.add_argument(
        "--snapshot_dir",
        type=str,
        default="snapshots_2D/UNet2D_dynamichead_logits_HATs/",
    )
    parser.add_argument(
        "--reload_path",
        type=str,
        default="snapshots_2D/UNet2D_dynamichead_logits/model.pth",
    )
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=False)
    parser.add_argument("--input_size", type=str, default="512,512")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--FP16", type=str2bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=101)
    parser.add_argument("--itrs_each_epoch", type=int, default=250)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--weight_std", type=str2bool, default=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--random_mirror", type=str2bool, default=True)
    parser.add_argument("--random_scale", type=str2bool, default=True)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--gpu", type=str, default="None")
    return parser


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, lr, num_stemps, power):
    lr = lr_poly(lr, i_iter, num_stemps, power)
    optimizer.param_groups[0]["lr"] = lr
    return lr


def count_score(preds, labels, rmin, rmax, cmin, cmax):
    Val_F1 = 0
    Val_DICE = 0
    Val_TPR = 0
    Val_PPV = 0
    cnt = 0

    for ki in range(len(preds)):
        cnt += 1
        pred = preds[ki, :, rmin:rmax, cmin:cmax]
        label = labels[ki, :, rmin:rmax, cmin:cmax]

        Val_DICE += dice_score(pred, label)
        preds1 = pred[1, ...].flatten().detach().cpu().numpy()
        labels1 = label[1, ...].detach().flatten().detach().cpu().numpy()

        cnf_matrix = confusion_matrix(preds1, labels1)

        try:
            FP = cnf_matrix[1, 0]
            FN = cnf_matrix[0, 1]
            TP = cnf_matrix[1, 1]
            TN = cnf_matrix[0, 0]
        except:
            FP = np.array(1)
            FN = np.array(1)
            TP = np.array(1)
            TN = np.array(1)

        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)

        Val_TPR += TP / (TP + FN)
        Val_PPV += TP / (TP + FP)

        Val_F1 += f1_score(preds1, labels1, average="macro")

    return Val_F1 / cnt, Val_DICE / cnt, Val_TPR / cnt, Val_PPV / cnt


def dice_score(preds, labels):  # on GPU
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

    dice = 2 * num / den
    return dice.mean()


def get_loss(images, preds, labels, weight, loss_seg_DICE, loss_seg_CE):
    term_seg_Dice = 0
    term_seg_BCE = 0
    term_all = 0

    term_seg_Dice += loss_seg_DICE.forward(preds, labels, weight)
    term_seg_BCE += loss_seg_CE.forward(preds, labels, weight)
    term_all += term_seg_Dice + term_seg_BCE

    return term_seg_Dice, term_seg_BCE, term_all


def supervise_learning(
    images,
    labels,
    batch_size,
    scales,
    model,
    now_task,
    weight,
    loss_seg_DICE,
    loss_seg_CE,
):
    preds = model(images, torch.ones(batch_size).cuda() * now_task, scales)
    labels = one_hot_3D(labels.long())

    term_seg_Dice, term_seg_BCE, term_all = get_loss(
        images, preds, labels, weight, loss_seg_DICE, loss_seg_CE
    )
    return term_seg_Dice, term_seg_BCE, term_all


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

        # Only use dice rather than bce in semi-supervised learning
        if now_relative == 1:
            # Background from this label should not have any overlap with the pred, --> 0
            semi_labels = 1 - labels
            semi_labels = one_hot_3D(semi_labels.long())
            semi_seg_Dice, semi_seg_BCE, semi_all = get_loss(
                images, semi_preds, semi_labels, weight, loss_seg_DICE, loss_seg_CE
            )
            term_seg_Dice -= semi_ratio * semi_seg_Dice * now_area_ratio
            term_all -= semi_ratio * semi_seg_Dice * now_area_ratio

        elif now_relative == -1:
            # Only supervised the regions which have label --> 1
            semi_labels = labels
            semi_preds = semi_labels.unsqueeze(1).repeat(1, 2, 1, 1) * semi_preds
            semi_labels = one_hot_3D(semi_labels.long())
            semi_seg_Dice, semi_seg_BCE, semi_all = get_loss(
                images, semi_preds, semi_labels, weight, loss_seg_DICE, loss_seg_CE
            )
            term_seg_Dice += semi_ratio * semi_seg_Dice * now_area_ratio
            term_all += semi_ratio * semi_seg_Dice * now_area_ratio

        elif now_relative == 2:
            # Foreground from this label should not have any overlap with the pred, --> 0
            semi_labels = labels
            semi_labels = one_hot_3D(semi_labels.long())
            semi_seg_Dice, semi_seg_BCE, semi_all = get_loss(
                images, semi_preds, semi_labels, weight, loss_seg_DICE, loss_seg_CE
            )
            term_seg_Dice -= semi_ratio * semi_seg_Dice * now_area_ratio
            term_all -= semi_ratio * semi_seg_Dice * now_area_ratio

    return term_seg_Dice, term_seg_BCE, term_all


def division_ratio(a, b):
    if a > b:
        return b / a
    else:
        return a / b


def main():
    start_time = timeit.default_timer()
    parser = get_arguments()
    print(parser)

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)

        writer = SummaryWriter(args.snapshot_dir)

        if not args.gpu == "None":
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        h, w = map(int, args.input_size.split(","))
        input_size = (h, w)

        cudnn.benchmark = True
        seed = args.random_seed
        if engine.distributed:
            seed = args.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Create model
        model = UNet2D_scale(num_classes=15, num_scale=4, weight_std=False)

        device = torch.device("cuda:{}".format(args.local_rank))
        model.to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), args.learning_rate, weight_decay=args.weight_decay
        )

        # Initialize AMP Scaler
        scaler = GradScaler(enabled=args.FP16)
        if args.FP16:
            print("Note: Using Native Torch AMP (FP16) during training************")

        if args.num_gpus > 1:
            model = engine.data_parallel(model)

        if args.reload_from_checkpoint:
            print("loading from checkpoint: {}".format(args.reload_path))
            if os.path.exists(args.reload_path):
                checkpoint = torch.load(
                    args.reload_path, map_location=torch.device("cpu")
                )
                model.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                if args.FP16 and "scaler" in checkpoint:
                    scaler.load_state_dict(checkpoint["scaler"])
                elif args.FP16 and "amp" in checkpoint:
                    print(
                        "Warning: Found legacy 'apex.amp' state in checkpoint. "
                        "Cannot load into 'torch.cuda.amp.GradScaler'. Starting scaler fresh."
                    )
            else:
                print("File not exists in the reload path: {}".format(args.reload_path))

        weights = [1.0, 1.0]
        loss_seg_DICE = loss.DiceLoss4MOTS(num_classes=args.num_classes).to(device)
        loss_seg_CE = loss.CELoss4MOTS(
            weight=weights, num_classes=args.num_classes, ignore_index=255
        ).to(device)

        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

        # Initialize Dataset and Sampler
        train_dataset = MOTSDataSet(
            args.trainset_dir,
            args.train_list,
            max_iters=args.itrs_each_epoch * args.batch_size,
            crop_size=input_size,
            scale=args.random_scale,
            mirror=args.random_mirror,
            edge_weight=args.edge_weight,
        )

        train_sampler = None
        if engine.distributed:
            train_sampler = DistributedSampler(train_dataset)

        trainloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.num_workers,
            sampler=train_sampler,
        )

        val_dataset = MOTSValDataSet(
            args.valset_dir,
            args.val_list,
            max_iters=args.itrs_each_epoch * args.batch_size,
            crop_size=input_size,
            scale=args.random_scale,
            mirror=args.random_mirror,
            edge_weight=args.edge_weight,
        )

        valloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        all_tr_loss_supervise = []
        all_tr_loss_all = []
        all_tr_loss = []

        layer_num = [0, 5, 12]
        semi_ratio = 0.1
        HATs_matrix = np.zeros((15, 15))

        Area = np.zeros((15))
        # ... [Keep Area initialization] ...
        Area[0] = 2.434
        Area[1] = 2.600
        Area[2] = 1.760
        Area[3] = 1.853
        Area[4] = 1.844
        Area[5] = 0.097
        Area[6] = 0.360
        Area[7] = 0.619
        Area[8] = 0.466
        Area[9] = 0.083
        Area[10] = 0.002
        Area[11] = 0.012
        Area[12] = 0.001
        Area[13] = 0.001
        Area[14] = 0.002

        Area_ratio = np.zeros((15, 15))
        for xi in range(0, 15):
            for yi in range(0, 15):
                Area_ratio[xi, yi] = division_ratio(Area[xi], Area[yi])

        # Matrix initialization
        # ... [Keep Matrix Initialization logic same as original] ...
        HATs_matrix[0, [1, 2, 3, 4, 7, 8, 10, 12, 13]] = 2
        HATs_matrix[0, 11] = 1
        HATs_matrix[1, 0] = 2
        HATs_matrix[1, [2, 3, 4, 7, 8, 10, 12, 13]] = 1
        HATs_matrix[1, 11] = 2
        for i in [2, 3, 4]:
            HATs_matrix[i, 0] = 2
            HATs_matrix[i, 1] = -1
            HATs_matrix[i, [j for j in [2, 3, 4] if j != i]] = 2
        HATs_matrix[5, [6, 7, 8, 9, 10, 11, 12, 13, 14]] = 2
        HATs_matrix[6, [5, 7, 8, 9, 10, 11, 12, 13, 14]] = 2
        HATs_matrix[7, [0, 5, 6, 9, 10, 11, 14]] = 2
        HATs_matrix[7, 1] = -1
        HATs_matrix[7, [8, 12, 13]] = 1
        HATs_matrix[8, [0, 5, 6, 9, 10, 11, 14]] = 2
        HATs_matrix[8, 1] = -1
        HATs_matrix[8, 7] = -1
        HATs_matrix[8, [12, 13]] = 1
        HATs_matrix[9, [5, 6, 7, 8, 10, 11, 12, 13]] = 2
        HATs_matrix[9, 14] = 1
        HATs_matrix[10, [0, 5, 6, 7, 8, 9, 11, 12, 13, 14]] = 2
        HATs_matrix[10, 1] = -1
        HATs_matrix[11, 0] = -1
        HATs_matrix[11, [1, 5, 6, 7, 8, 9, 10, 12, 13, 14]] = 2
        HATs_matrix[12, [0, 5, 6, 9, 10, 11, 13, 14]] = 2
        HATs_matrix[12, 1] = -1
        HATs_matrix[12, [7, 8]] = -1
        HATs_matrix[13, [0, 5, 6, 9, 10, 11, 12, 14]] = 2
        HATs_matrix[13, 1] = -1
        HATs_matrix[13, [7, 8]] = -1
        HATs_matrix[14, [5, 6, 7, 8, 10, 11, 12, 13]] = 2
        HATs_matrix[14, 9] = -1

        df_loss = pd.DataFrame(
            columns=["epoch", "epoch_loss_supervise_mean", "semi_all"]
        )

        for epoch in range(args.start_epoch, args.num_epochs):
            model.train()

            # Initialize ImagePools for all 15 tasks
            task_pools = {
                i: {
                    "image": ImagePool(8),
                    "mask": ImagePool(8),
                    "weight": ImagePool(8),
                    "scale": [],
                    "layer": [],
                }
                for i in range(15)
            }

            if engine.distributed:
                train_sampler.set_epoch(epoch)

            epoch_loss = []
            adjust_learning_rate(
                optimizer, epoch, args.learning_rate, args.num_epochs, args.power
            )

            task_num = 15
            each_loss = torch.zeros((task_num)).cuda()
            count_batch = torch.zeros((task_num)).cuda()
            supervised_loss = torch.zeros((task_num)).cuda()

            for iter, batch in enumerate(trainloader):
                imgs = batch[0].cuda()
                lbls = batch[1].cuda()
                wt = batch[2].cuda().float()
                # batch[3] is name, skipped
                l_ids = batch[4].cuda()
                t_ids = batch[5].cuda()
                s_ids = batch[6].cuda()

                for ki in range(len(imgs)):
                    now_task = layer_num[l_ids[ki]] + t_ids[ki]
                    now_task_item = now_task.item()

                    if now_task_item in task_pools:
                        pool = task_pools[now_task_item]
                        pool["image"].add(imgs[ki].unsqueeze(0))
                        pool["mask"].add(lbls[ki].unsqueeze(0))
                        pool["weight"].add(wt[ki].unsqueeze(0))
                        pool["scale"].append(s_ids[ki])
                        pool["layer"].append(l_ids[ki])

                for t_idx in range(15):
                    pool = task_pools[t_idx]
                    if pool["image"].num_imgs >= args.batch_size:
                        images = pool["image"].query(args.batch_size)
                        labels = pool["mask"].query(args.batch_size)
                        wts = pool["weight"].query(args.batch_size)
                        scales = torch.ones(args.batch_size).cuda()
                        for bi in range(len(scales)):
                            scales[bi] = pool["scale"].pop(0)

                        now_task = t_idx
                        weight = args.edge_weight**wts

                        # --- UPDATED TRAINING STEP ---
                        with autocast(enabled=args.FP16):
                            term_seg_Dice, term_seg_BCE, Sup_term_all = (
                                supervise_learning(
                                    images,
                                    labels,
                                    args.batch_size,
                                    scales,
                                    model,
                                    now_task,
                                    weight,
                                    loss_seg_DICE,
                                    loss_seg_CE,
                                )
                            )

                            term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(
                                images,
                                labels,
                                args.batch_size,
                                scales,
                                model,
                                now_task,
                                weight,
                                loss_seg_DICE,
                                loss_seg_CE,
                                term_seg_Dice,
                                term_seg_BCE,
                                Sup_term_all,
                                HATs_matrix,
                                semi_ratio,
                                Area_ratio,
                            )

                        reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                        reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                        reduce_all = engine.all_reduce_tensor(All_term_all)

                        optimizer.zero_grad()
                        # Use Scaler for backward
                        scaler.scale(reduce_all).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        if iter % 50 == 0:
                            print(
                                "Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}".format(
                                    epoch,
                                    iter,
                                    len(trainloader),
                                    optimizer.param_groups[0]["lr"],
                                    reduce_Dice.item(),
                                    reduce_BCE.item(),
                                    reduce_all.item(),
                                )
                            )

                        supervise_all = engine.all_reduce_tensor(Sup_term_all)
                        supervised_loss[now_task] += supervise_all
                        each_loss[now_task] += reduce_all
                        count_batch[now_task] += 1
                        epoch_loss.append(float(reduce_all))

            # Last round clean up
            for t_idx in range(15):
                pool = task_pools[t_idx]
                if pool["image"].num_imgs > 0:
                    current_batch_size = pool["image"].num_imgs
                    images = pool["image"].query(current_batch_size)
                    labels = pool["mask"].query(current_batch_size)
                    wts = pool["weight"].query(current_batch_size)
                    scales = torch.ones(current_batch_size).cuda()
                    for bi in range(len(scales)):
                        scales[bi] = pool["scale"].pop(0)

                    now_task = t_idx
                    weight = args.edge_weight**wts

                    with autocast(enabled=args.FP16):
                        term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(
                            images,
                            labels,
                            current_batch_size,
                            scales,
                            model,
                            now_task,
                            weight,
                            loss_seg_DICE,
                            loss_seg_CE,
                        )
                        term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(
                            images,
                            labels,
                            current_batch_size,
                            scales,
                            model,
                            now_task,
                            weight,
                            loss_seg_DICE,
                            loss_seg_CE,
                            term_seg_Dice,
                            term_seg_BCE,
                            Sup_term_all,
                            HATs_matrix,
                            semi_ratio,
                            Area_ratio,
                        )

                    reduce_all = engine.all_reduce_tensor(All_term_all)
                    optimizer.zero_grad()
                    scaler.scale(reduce_all).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    supervise_all = engine.all_reduce_tensor(Sup_term_all)
                    supervised_loss[now_task] += supervise_all
                    each_loss[now_task] += reduce_all
                    count_batch[now_task] += 1
                    epoch_loss.append(float(reduce_all))

            epoch_loss = np.mean(epoch_loss)
            supervised_loss_val = np.mean(supervised_loss.detach().cpu().numpy())
            print(
                "Loss Summary: Epoch={}, Total Loss={:.4f}, Supervised Loss={:.4f}".format(
                    epoch, epoch_loss, supervised_loss_val
                )
            )

            all_tr_loss_supervise.append(supervised_loss_val)
            all_tr_loss_all.append(epoch_loss)
            all_tr_loss.append(epoch_loss)

            if args.local_rank == 0:
                writer.add_scalar(
                    "learning_rate", optimizer.param_groups[0]["lr"], epoch
                )
                writer.add_scalar("Train_loss", epoch_loss.item(), epoch)

                plt.plot(all_tr_loss_supervise, label="Supervise")
                plt.plot(all_tr_loss_all, label="Supervise + Psuedo")
                plt.legend()
                plt.savefig(
                    "TrainingLoss_%s.png" % (os.path.basename(args.snapshot_dir))
                )
                plt.clf()

                row = len(df_loss)
                df_loss.loc[row] = [epoch, supervised_loss_val, epoch_loss]
                df_loss.to_csv("TrainingLoss_HATs.csv")

            # Validation Loop
            if (
                epoch >= 0
                and args.local_rank == 0
                and (epoch % 10 == 0 or epoch == args.num_epochs - 1)
            ):
                print("Starting validation...")
                model.eval()
                val_pools = {
                    i: {"image": ImagePool(8), "mask": ImagePool(8), "scale": []}
                    for i in range(15)
                }

                val_metrics = np.zeros((5, 15))

                with torch.no_grad():
                    for batch1 in valloader:
                        imgs = batch1[0].cuda()
                        lbls = batch1[1].cuda()
                        l_ids = batch1[4].cuda()
                        t_ids = batch1[5].cuda()
                        s_ids = batch1[6].cuda()

                        for ki in range(len(imgs)):
                            now_task = (layer_num[l_ids[ki]] + t_ids[ki]).item()
                            if now_task in val_pools:
                                val_pools[now_task]["image"].add(imgs[ki].unsqueeze(0))
                                val_pools[now_task]["mask"].add(lbls[ki].unsqueeze(0))
                                val_pools[now_task]["scale"].append(s_ids[ki])

                        for t_idx in range(15):
                            pool = val_pools[t_idx]
                            while pool["image"].num_imgs >= args.batch_size:
                                images = pool["image"].query(args.batch_size)
                                labels = pool["mask"].query(args.batch_size)
                                scales = torch.ones(args.batch_size).cuda()
                                for bi in range(len(scales)):
                                    scales[bi] = pool["scale"].pop(0)

                                # Validation runs in FP32 usually, but ok to use autocast
                                # here we skip autocast for val unless memory is tight.
                                if t_idx <= 4:
                                    preds = torch.zeros(
                                        (args.batch_size, 2, 1024, 1024)
                                    ).cuda()
                                    crops = [
                                        (0, 512, 0, 512),
                                        (0, 512, 512, 1024),
                                        (512, 1024, 512, 1024),
                                        (512, 1024, 0, 512),
                                    ]
                                    for r1, r2, c1, c2 in crops:
                                        preds[:, :, r1:r2, c1:c2] = model(
                                            images[:, :, r1:r2, c1:c2],
                                            torch.ones(args.batch_size).cuda() * t_idx,
                                            scales,
                                        )
                                    rmin, rmax, cmin, cmax = 0, 1024, 0, 1024
                                else:
                                    preds = model(
                                        images[:, :, 256:768, 256:768],
                                        torch.ones(args.batch_size).cuda() * t_idx,
                                        scales,
                                    )
                                    if t_idx <= 10:
                                        rmin, rmax, cmin, cmax = 128, 384, 128, 384
                                    else:
                                        rmin, rmax, cmin, cmax = 0, 512, 0, 512

                                now_preds = torch.argmax(preds, 1) == 1
                                now_preds_onehot = one_hot_3D(now_preds.long())
                                if t_idx <= 4:
                                    labels_onehot = one_hot_3D(labels.long())
                                else:
                                    labels_onehot = one_hot_3D(
                                        labels[:, 256:768, 256:768].long()
                                    )

                                F1, DICE, TPR, PPV = count_score(
                                    now_preds_onehot,
                                    labels_onehot,
                                    rmin,
                                    rmax,
                                    cmin,
                                    cmax,
                                )
                                val_metrics[0, t_idx] += F1
                                val_metrics[1, t_idx] += DICE
                                val_metrics[2, t_idx] += TPR
                                val_metrics[3, t_idx] += PPV
                                val_metrics[4, t_idx] += 1

                    # Clean up (remaining validation batches)
                    for t_idx in range(15):
                        pool = val_pools[t_idx]
                        if pool["image"].num_imgs > 0:
                            current_batch_size = pool["image"].num_imgs
                            images = pool["image"].query(current_batch_size)
                            labels = pool["mask"].query(current_batch_size)
                            scales = torch.ones(current_batch_size).cuda()
                            for bi in range(len(scales)):
                                scales[bi] = pool["scale"].pop(0)

                            if t_idx <= 4:
                                preds = torch.zeros(
                                    (current_batch_size, 2, 1024, 1024)
                                ).cuda()
                                crops = [
                                    (0, 512, 0, 512),
                                    (0, 512, 512, 1024),
                                    (512, 1024, 512, 1024),
                                    (512, 1024, 0, 512),
                                ]
                                for r1, r2, c1, c2 in crops:
                                    preds[:, :, r1:r2, c1:c2] = model(
                                        images[:, :, r1:r2, c1:c2],
                                        torch.ones(current_batch_size).cuda() * t_idx,
                                        scales,
                                    )
                                rmin, rmax, cmin, cmax = 0, 1024, 0, 1024
                                labels_onehot = one_hot_3D(labels.long())
                            else:
                                preds = model(
                                    images[:, :, 256:768, 256:768],
                                    torch.ones(current_batch_size).cuda() * t_idx,
                                    scales,
                                )
                                if t_idx <= 10:
                                    rmin, rmax, cmin, cmax = 128, 384, 128, 384
                                else:
                                    rmin, rmax, cmin, cmax = 0, 512, 0, 512
                                labels_onehot = one_hot_3D(
                                    labels[:, 256:768, 256:768].long()
                                )

                            now_preds = torch.argmax(preds, 1) == 1
                            now_preds_onehot = one_hot_3D(now_preds.long())
                            F1, DICE, TPR, PPV = count_score(
                                now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax
                            )
                            val_metrics[0, t_idx] += F1
                            val_metrics[1, t_idx] += DICE
                            val_metrics[2, t_idx] += TPR
                            val_metrics[3, t_idx] += PPV
                            val_metrics[4, t_idx] += 1

                    if engine.distributed:
                        val_metrics_tensor = torch.tensor(val_metrics).float().cuda()
                        val_metrics_tensor = engine.all_reduce_tensor(
                            val_metrics_tensor, norm=False
                        )
                        val_metrics = val_metrics_tensor.cpu().numpy()

                    cnt = val_metrics[4, :] + 1e-6
                    avg_F1 = val_metrics[0, :] / cnt
                    avg_Dice = val_metrics[1, :] / cnt
                    avg_TPR = val_metrics[2, :] / cnt
                    avg_PPV = val_metrics[3, :] / cnt

                    df_val = pd.DataFrame(
                        {
                            "Task": range(15),
                            "F1": avg_F1,
                            "Dice": avg_Dice,
                            "TPR": avg_TPR,
                            "PPV": avg_PPV,
                        }
                    )
                    print("Validation Results:\n", df_val)
                    df_val.to_csv(
                        os.path.join(args.snapshot_dir, f"val_results_e{epoch}.csv")
                    )

                # Save Model
                print("Saving model...")
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    # Saving Scaler State instead of Amp
                    "scaler": scaler.state_dict() if args.FP16 else None,
                }
                torch.save(
                    state, osp.join(args.snapshot_dir, f"UNet2D_DynConv_e{epoch}.pth")
                )

        end_time = timeit.default_timer()
        print("Total training time: {:.2f} seconds".format(end_time - start_time))


if __name__ == "__main__":
    main()
