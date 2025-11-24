import os
import glob
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import the model definition from your provided file
try:
    from unet2D_Dodnet_scale_token import unet2D
except ImportError:
    print(
        "Error: 'unet2D_Dodnet_scale_token.py' not found. Please ensure it is in the same directory."
    )
    exit(1)


def get_args():
    parser = argparse.ArgumentParser(
        description="Test PrPSeg 8-class Model on Custom Dataset"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=".",
        help="Root directory containing data folders",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="model_weights/Omni-Seg_cls_sls_token_PrPSeg_8class.pth",
        help="Path to pretrained weights",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_results",
        help="Directory to save results",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    return parser.parse_args()


def get_class_info(folder_name):
    """
    Maps folder names to Task ID (Class) and Scale ID based on PrPSeg 8-class specifications.
    Returns: (task_id, scale_id, class_name) or (None, None, None) if skipped.
    """
    lower_name = folder_name.lower()

    # Exclusion list (Classes not in the 8-class model)
    if "vessels" in lower_name or "ptc" in lower_name:
        return None, None, None

    # Mapping logic
    # Scale 0: 5x, Scale 1: 10x, Scale 2: 20x, Scale 3: 40x

    # 1. Functional Units
    if "dt" in lower_name:  # Distal Tubule
        return 2, 1, "DT"
    elif "pt" in lower_name:  # Proximal Tubule
        return 3, 1, "PT"
    elif "capsule" in lower_name or "cap" in lower_name:  # Glomerular Capsule
        return 4, 0, "Capsule"
    elif "tuft" in lower_name:  # Glomerular Tuft
        return 5, 0, "Tuft"

    # 2. Regions (Less likely to be in patch folders, but included for completeness)
    elif "medulla" in lower_name:
        return 0, 0, "Medulla"
    elif "cortex" in lower_name:
        return 1, 0, "Cortex"

    # 3. Cells
    elif "pod" in lower_name:
        return 6, 2, "Podocyte"
    elif "mes" in lower_name:
        return 7, 2, "Mesangial"

    return None, None, None


def main():
    args = get_args()

    # Device configuration
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # 1. Initialize Model
    # PrPSeg 8-class configuration: num_classes=8, num_scale=4
    print("Initializing UNet2D (PrPSeg 8-class)...")
    # Note: We instantiate unet2D class directly to pass specific args
    model = unet2D(layers=[1, 2, 2, 2, 2], num_classes=8, num_scale=4, weight_std=False)

    # 2. Load Weights
    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found at {args.weights}")
        return

    print(f"Loading weights from {args.weights}...")
    checkpoint = torch.load(args.weights, map_location="cpu")

    # Handle potential dictionary wrapping in checkpoint
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Clean up state_dict keys if necessary (e.g. remove 'module.' prefix)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v

    try:
        model.load_state_dict(new_state_dict)
    except Exception as e:
        print(f"Warning loading weights: {e}")
        print("Attempting strict=False load...")
        model.load_state_dict(new_state_dict, strict=False)

    model.to(device)
    model.eval()

    # 3. Prepare Data Transformations
    # Resize to 512x512 (standard for this model architecture) and normalize
    transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            # Standard normalization often helps, though strictly 0-1 might work too depending on training
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 4. Inference Loop
    dirs = [
        d
        for d in os.listdir(args.data_root)
        if os.path.isdir(os.path.join(args.data_root, d))
    ]

    for d in sorted(dirs):
        dir_path = os.path.join(args.data_root, d)

        # Determine class and scale from folder name
        task_id, scale_id, class_name = get_class_info(d)

        if task_id is None:
            print(
                f"Skipping folder '{d}' (Class not found in 8-class model or explicitly excluded)"
            )
            continue

        print(
            f"\nProcessing folder: {d} | Target: {class_name} (ID: {task_id}) | Scale ID: {scale_id}"
        )

        # Create output directory
        save_path = os.path.join(args.output_dir, d)
        os.makedirs(save_path, exist_ok=True)

        images = (
            glob.glob(os.path.join(dir_path, "*.png"))
            + glob.glob(os.path.join(dir_path, "*.jpg"))
            + glob.glob(os.path.join(dir_path, "*.jpeg"))
        )

        if not images:
            print(f"No images found in {dir_path}")
            continue

        with torch.no_grad():
            for img_path in tqdm(images):
                img_name = os.path.basename(img_path)

                try:
                    # Load and Preprocess
                    img_pil = Image.open(img_path).convert("RGB")
                    original_size = img_pil.size
                    img_tensor = (
                        transform(img_pil).unsqueeze(0).to(device)
                    )  # (1, 3, 512, 512)

                    # Prepare task and scale tensors
                    task_tensor = torch.tensor([task_id], dtype=torch.float32).to(
                        device
                    )
                    scale_tensor = torch.tensor([scale_id], dtype=torch.float32).to(
                        device
                    )

                    # Forward Pass
                    # logits shape: (1, 2, 512, 512) -> 2 channels (background, foreground)
                    logits = model(img_tensor, task_tensor, scale_tensor)

                    # Generate Prediction
                    # Argmax to get class 0 or 1
                    pred = (
                        torch.argmax(logits, dim=1)
                        .squeeze(0)
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                    )

                    # Post-processing for visualization/saving
                    # 1. Resize back to original size if necessary (Optional, typically we verify on patches)
                    # Here we save the 512x512 result or you can use cv2.resize to put it back

                    # Save binary mask (multiply by 255 to make it visible)
                    pred_img = Image.fromarray(pred * 255)
                    pred_img.save(
                        os.path.join(save_path, img_name.replace(".", "_mask."))
                    )

                    # Optional: Save overlay
                    plt.imsave(
                        os.path.join(save_path, img_name.replace(".", "_overlay.")),
                        pred,
                        cmap="gray",
                    )

                except Exception as e:
                    print(f"Failed to process {img_name}: {e}")

    print(f"\nProcessing complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
