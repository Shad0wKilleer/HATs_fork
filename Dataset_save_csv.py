import os
import glob
import pandas as pd


def create_data_csv(root):
    """
    Scans a directory with the HATs project's specific structure,
    pairs images (png/tif) with their corresponding masks,
    and saves all file paths to 'data_list.csv' in the root directory.
    """
    layer_list = []
    task_list = []
    scale_list = []
    image_path_list = []
    label_path_list = []

    print("Scanning for class folders...")
    # Find all class folders (e.g., '0_0_0_medulla', '1_0_1_dt')
    tasks = glob.glob(os.path.join(root, "*"))

    for task_path in tasks:
        if not os.path.isdir(task_path):
            continue

        task_basename = os.path.basename(task_path)
        try:
            # Parse folder name like '1_0_1_dt'
            layer_id = task_basename.split("_")[0]
            tasks_id = task_basename.split("_")[1]
            scale_id = task_basename.split("_")[2]
        except IndexError:
            print(f"--> Skipping folder with unexpected name: {task_basename}")
            continue

        # Find all stain folders (e.g., 'HE', 'PAS')
        stain_folders = glob.glob(os.path.join(task_path, "*"))

        for stain_path in stain_folders:
            if not os.path.isdir(stain_path):
                continue

            # Find all files in the stain folder
            all_files = glob.glob(os.path.join(stain_path, "*"))

            # Find all image files (png or tif) that are NOT masks
            image_files = [
                f
                for f in all_files
                if (f.endswith(".png") or f.endswith(".tif"))
                and "mask" not in f
                and "lbl" not in f
            ]

            if not image_files:
                continue

            for image_root in image_files:
                # Find the corresponding mask
                base_name, ext = os.path.splitext(os.path.basename(image_root))

                # Handle names like 'im_100_img.png'
                if "_img" in base_name:
                    base_name = base_name.replace("_img", "")

                mask_search_pattern = os.path.join(stain_path, f"{base_name}_mask*")
                mask_files = glob.glob(mask_search_pattern)

                if not mask_files:
                    print(f"--> Warning: No mask found for image: {image_root}")
                    continue

                mask_root = mask_files[0]  # Take the first matching mask

                # Add the paired paths to our lists
                layer_list.append(int(layer_id))
                task_list.append(int(tasks_id))
                scale_list.append(int(scale_id))
                image_path_list.append(image_root)
                label_path_list.append(mask_root)

    if not image_path_list:
        print("Error: No images found. Check your directory structure.")
        return

    # --- Create and save the DataFrame ---
    df = pd.DataFrame(
        columns=["image_path", "label_path", "name", "layer_id", "task_id", "scale_id"]
    )

    print(f"\nFound {len(image_path_list)} image/mask pairs. Creating DataFrame...")
    for i in range(len(image_path_list)):
        image_path = image_path_list[i]
        label_path = label_path_list[i]

        # Create a unique name from the path, e.g., "-1_0_1_dt-HE-im_100.png"
        name = image_path.replace(root, "").replace(os.path.sep, "-")

        df.loc[i] = [
            image_path,
            label_path,
            name,
            layer_list[i],
            task_list[i],
            scale_list[i],
        ]

    output_path = os.path.join(root, "data_list.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSuccess! CSV file saved to: {output_path}")


if __name__ == "__main__":
    trainset_dir = "/content/drive/My Drive/Renal Segmentation/HATs_data/val"

    print(f"Starting script. Scanning directory: {trainset_dir}")
    create_data_csv(root=trainset_dir)
    print("Script finished.")
