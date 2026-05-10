import pandas as pd
import numpy as np


def select_top_moas(input_file_path, output_file_path='gigadb_top_30_moas.csv', top_n=30):
    """
    Select the top N MoA columns that have the most 1s and -1s combined.

    Parameters:
    input_file_path (str): Path to the input CSV file
    output_file_path (str): Path where the filtered CSV will be saved
    top_n (int): Number of top MoA columns to select
    """
    # Read the CSV file
    print(f"Reading data from {input_file_path}...")
    df = pd.read_csv(input_file_path)

    # Metadata columns that should remain unchanged
    metadata_columns = [
        'image_id', 'plate_id', 'well_id', 'site_id', 'BROAD_ID',
        'canonical_smiles', 'chembl_id', 'hier_split',
        'hier_split_0', 'hier_split_1', 'hier_split_2', 'hier_split_3', 'hier_split_4'
    ]

    # Get all MoA columns (all columns except metadata)
    moa_columns = [col for col in df.columns if col not in metadata_columns]

    print(f"Total number of compounds: {len(df)}")
    print(f"Total number of MoA tasks: {len(moa_columns)}")

    # Count 1s and -1s for each MoA column
    moa_counts = {}
    for col in moa_columns:
        active_count = (df[col] == 1).sum()
        inactive_count = (df[col] == -1).sum()
        total_labels = active_count + inactive_count
        moa_counts[col] = {
            'active': active_count,
            'inactive': inactive_count,
            'total': total_labels
        }

    # Sort MoA columns by total number of labels (1s and -1s)
    sorted_moas = sorted(moa_counts.items(), key=lambda x: x[1]['total'], reverse=True)

    # Select top N MoAs
    top_moas = [moa[0] for moa in sorted_moas[:top_n]]

    print(f"\nSelected top {top_n} MoAs with most labels:")
    for i, moa in enumerate(top_moas, 1):
        counts = moa_counts[moa]
        print(f"{i}. {moa}: {counts['total']} total labels ({counts['active']} active, {counts['inactive']} inactive)")

    # Create a new dataframe with metadata columns and top MoAs
    columns_to_keep = metadata_columns + top_moas
    df_filtered = df[columns_to_keep]

    # Rename MoA columns to add 'moa_' prefix
    rename_dict = {col: f"moa_{col}" for col in top_moas}
    df_filtered = df_filtered.rename(columns=rename_dict)

    # Save the filtered data
    df_filtered.to_csv(output_file_path, index=False)
    print(f"\nFiltered data with top {top_n} MoAs saved to {output_file_path}")

    return df_filtered


# Example usage
if __name__ == "__main__":
    input_path = "/Users/jkosciukiewicz/Developer/UJ/Projects/HCS/data/gigadb/gigadb_MoA_with_images_filtered_25.csv"
    output_path = "/Users/jkosciukiewicz/Developer/UJ/Projects/HCS/data/gigadb/gigadb_top_30_moas.csv"

    filtered_data = select_top_moas(input_path, output_path, top_n=30)