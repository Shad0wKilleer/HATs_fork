import torch
import matplotlib.pyplot as plt
from custom_dataset import RenalDataset

# Define paths to your OVERFIT CSVs
csvs = ["dataset_pt_overfit.csv", "dataset_dt_overfit.csv"]

# Load Dataset
ds = RenalDataset(csvs, is_training=True, crop_size=(512, 512))

# Get one sample
img, mask, task_id, _ = ds[0]

# Un-normalize for display
# Image is (C, H, W) float 0-1. Convert to (H, W, C)
img_disp = img.permute(1, 2, 0).numpy()

print(f"Image Shape: {img.shape}, Max: {img.max()}, Min: {img.min()}")
print(f"Mask Shape: {mask.shape}, Unique Values: {torch.unique(mask)}")
print(f"Task ID: {task_id}")

# Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img_disp)
ax[0].set_title(f"Input Image (Task {task_id})")
ax[1].imshow(mask, cmap="gray")
ax[1].set_title("Target Mask")
plt.savefig("debug_input.png")
print("Saved 'debug_input.png'. Check this file!")
