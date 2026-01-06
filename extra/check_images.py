# This script loads the images to see if any is truncated or not. If any is, it
# will simply output it so we can delete it manually.
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import argparse


def check_files(csv_path):
    print(f"Checking dataset list: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    corrupt_files = []

    # Iterate over every row in the CSV
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Verifying images"):
        # We check both image and label
        paths_to_check = [row["image_path"], row["label_path"]]

        for file_path in paths_to_check:
            # Check if file exists first
            if not os.path.exists(file_path):
                print(f"[MISSING] {file_path}")
                corrupt_files.append(file_path)
                continue

            # Check for corruption/truncation
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Verify file integrity (fast)
            except (IOError, SyntaxError) as e:
                print(f"[CORRUPT] {file_path} -> {e}")
                corrupt_files.append(file_path)
            except Exception as e:
                print(f"[ERROR] {file_path} -> {e}")
                corrupt_files.append(file_path)

    if corrupt_files:
        print("\nFound the following corrupt/missing files:")
        for f in set(corrupt_files):  # set to avoid duplicates
            print(f)
        print(f"\nTotal corrupt files: {len(set(corrupt_files))}")
        print("Please delete these files or remove their rows from the CSV.")
    else:
        print("\nAll files verified successfully! No corruption found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Using the defaults you had in your train script
    parser.add_argument(
        "--trainset_dir",
        type=str,
        default="/home/shad0wkillar/programming/csv_files/og_code/data_list_train.csv",
    )
    parser.add_argument(
        "--valset_dir",
        type=str,
        default="/home/shad0wkillar/programming/csv_files/og_code/data_list_test.csv",
    )
    args = parser.parse_args()

    check_files(args.trainset_dir)
    print("-" * 50)
    check_files(args.valset_dir)
