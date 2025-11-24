import os
import glob
import argparse
import torch
import shutil
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Import the architecture (Essential for loading weights)
try:
    from unet2D_Dodnet_scale_token import unet2D
except ImportError:
    print(
        "Error: 'unet2D_Dodnet_scale_token.py' not found. Please ensure it is in the same directory."
    )
    exit(1)


def get_args():
    parser = argparse.ArgumentParser(description="Test PrPSeg & Sort by Dice Score")

    # Path Arguments
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Root folder containing data subfolders",
    )
    parser.add_argument(
        "--weights_path", type=str, required=True, help="Path to .pth model weights"
    )
    parser.add_argument(
        "--output_folder", type=str, required=True, help="Path to save sorted results"
    )

    # Settings
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--mask_suffix",
        type=str,
        default="_mask",
        help="Suffix for ground truth files (e.g. '_mask' for image_01_mask.png)",
    )

    return parser.parse_args()


def get_class_info(folder_name):
    """Maps folder names to PrPSeg 8-class ID and Scale."""
    lower = folder_name.lower()

    # Exclude irrelevant folders
    if "vessels" in lower or "ptc" in lower:
        return None, None

    # Mapping
    if "dt" in lower:
        return 2, 1  # Distal Tubule
    elif "pt" in lower:
        return 3, 1  # Proximal Tubule
    elif "capsule" in lower:
        return 4, 0  # Capsule
    elif "tuft" in lower:
        return 5, 0  # Tuft
    elif "medulla" in lower:
        return 0, 0
    elif "cortex" in lower:
        return 1, 0
    elif "pod" in lower:
        return 6, 2
    elif "mes" in lower:
        return 7, 2

    return None, None


def calculate_dice(pred, gt):
    """Calculates binary Dice score (0.0 to 1.0)."""
    # Flatten
    pred = pred.flatten()
    gt = gt.flatten()

    # Ensure binary
    pred = (pred > 0).astype(np.float32)
    gt = (gt > 0).astype(np.float32)

    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return (2.0 * intersection) / union


def get_score_folder(dice_score):
    """Returns folder name based on Dice score percentage."""
    score_pct = dice_score * 100

    if score_pct <= 10:
        return "Score_00-10"
    elif score_pct <= 20:
        return "Score_11-20"
    elif score_pct <= 30:
        return "Score_21-30"
    elif score_pct <= 40:
        return "Score_31-40"
    elif score_pct <= 50:
        return "Score_41-50"
    elif score_pct <= 60:
        return "Score_51-60"
    elif score_pct <= 70:
        return "Score_61-70"
    elif score_pct <= 80:
        return "Score_71-80"
    elif score_pct <= 90:
        return "Score_81-90"
    else:
        return "Score_91-100"


def save_overlay(original_pil, pred_mask, save_path):
    """Saves image with red mask overlay."""
    mask_rgb = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    mask_rgb[pred_mask == 1] = [255, 0, 0]

    mask_pil = Image.fromarray(mask_rgb).convert("RGBA")
    original_rgba = original_pil.convert("RGBA")

    alpha = np.zeros_like(pred_mask, dtype=np.uint8)
    alpha[pred_mask == 1] = 100
    mask_pil.putalpha(Image.fromarray(alpha))

    overlay = Image.alpha_composite(original_rgba, mask_pil)
    overlay.save(save_path)


def main():
    args = get_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # --- 1. Load Model ---
    print("Loading Model...")
    model = unet2D(layers=[1, 2, 2, 2, 2], num_classes=8, num_scale=4, weight_std=False)

    checkpoint = torch.load(args.weights_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    # Fix 'module.' keys if present
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    try:
        model.load_state_dict(new_state_dict)
    except:
        model.load_state_dict(new_state_dict, strict=False)

    model.to(device)
    model.eval()

    # --- 2. Setup Transform ---
    transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # --- 3. Process Images ---
    dirs = [
        d
        for d in os.listdir(args.data_path)
        if os.path.isdir(os.path.join(args.data_path, d))
    ]

    for d in sorted(dirs):
        task_id, scale_id = get_class_info(d)
        if task_id is None:
            continue  # Skip ignored folders

        print(f"\nProcessing Folder: {d} (Class ID: {task_id})")
        input_dir = os.path.join(args.data_path, d)

        # Find images (exclude masks)
        all_files = glob.glob(os.path.join(input_dir, "*.*"))
        images = [
            f
            for f in all_files
            if f.lower().endswith((".png", ".jpg")) and args.mask_suffix not in f
        ]

        with torch.no_grad():
            for img_path in tqdm(images):
                filename = os.path.splitext(os.path.basename(img_path))[0]

                try:
                    # Load Image
                    img_pil = Image.open(img_path).convert("RGB")
                    w, h = img_pil.size

                    # Predict
                    img_t = transform(img_pil).unsqueeze(0).to(device)
                    task_t = torch.tensor([task_id], dtype=torch.float32).to(device)
                    scale_t = torch.tensor([scale_id], dtype=torch.float32).to(device)

                    logits = model(img_t, task_t, scale_t)
                    pred = (
                        torch.argmax(logits, dim=1)
                        .squeeze(0)
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                    )

                    # Resize Pred to Original Size
                    pred_pil = Image.fromarray(pred).resize(
                        (w, h), resample=Image.NEAREST
                    )
                    pred_final = np.array(pred_pil)

                    # Check for Ground Truth
                    gt_path = os.path.join(
                        input_dir, f"{filename}{args.mask_suffix}.png"
                    )

                    if os.path.exists(gt_path):
                        # Load GT
                        gt_pil = Image.open(gt_path).convert(
                            "L"
                        )  # Convert to grayscale/index
                        gt_pil = gt_pil.resize((w, h), resample=Image.NEAREST)
                        gt_arr = np.array(gt_pil)

                        # Threshold GT (assuming binary mask 0 or 255)
                        gt_final = (gt_arr > 0).astype(np.uint8)

                        # Calculate Score
                        score = calculate_dice(pred_final, gt_final)
                        folder_name = get_score_folder(score)

                    else:
                        folder_name = "No_GroundTruth"
                        gt_path = None  # Marker that no GT exists

                    # Create Save Directory
                    save_dir = os.path.join(args.output_folder, folder_name)
                    os.makedirs(save_dir, exist_ok=True)

                    # Save Files
                    # 1. Original
                    shutil.copy(
                        img_path, os.path.join(save_dir, f"{filename}_orig.png")
                    )

                    # 2. Predicted Mask
                    Image.fromarray(pred_final * 255).save(
                        os.path.join(save_dir, f"{filename}_pred.png")
                    )

                    # 3. Overlay
                    save_overlay(
                        img_pil,
                        pred_final,
                        os.path.join(save_dir, f"{filename}_overlay.png"),
                    )

                    # 4. Ground Truth (if exists)
                    if gt_path:
                        shutil.copy(
                            gt_path, os.path.join(save_dir, f"{filename}_gt.png")
                        )

                except Exception as e:
                    print(f"Failed {filename}: {e}")

    print(f"\nProcessing Complete. Results sorted in: {args.output_folder}")


if __name__ == "__main__":
    main()
