"""Utility script to merge heterogeneous hardware datasets (iGPU + dGPU)."""

import os
import argparse
import pandas as pd

def merge_datasets(input_files, output_file):
    dfs = []
    for f in input_files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            print(f"Loaded {len(df)} rows from {f}")
            dfs.append(df)
        else:
            print(f"Warning: File {f} not found, skipping.")

    if not dfs:
        print("Error: No datasets found to merge.")
        return

    merged_df = pd.concat(dfs, ignore_index=True)
    # Remove any potential duplicate benchmark configurations
    merged_df = merged_df.drop_duplicates()

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    merged_df.to_csv(output_file, index=False)
    print(f"\nSuccessfully created merged dataset with {len(merged_df)} total rows: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple hardware benchmark CSV datasets.")
    parser.add_argument("--inputs", nargs="+", default=["dataset/dataset.csv", "dataset/colab_dataset.csv"], help="List of input CSV files.")
    parser.add_argument("--output", default="dataset/dataset_merged.csv", help="Merged output CSV file path.")
    args = parser.parse_args()

    merge_datasets(args.inputs, args.output)
