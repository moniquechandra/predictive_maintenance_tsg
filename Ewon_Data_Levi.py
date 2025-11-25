import numpy as np
import pandas as pd
import glob

def concat_csv_files():
    try:
        existing = pd.read_feather("data/New_SVRM3_Ewon/combined.feather")
        return existing
    
    except:
        csv_files = glob.glob("data/New_SVRM3_Ewon/*.csv")
        def load_csv(path):
            df = pd.read_csv(path, sep=";",dtype=str)
            print(path, df.shape)
            return df
        dfs = [load_csv(path) for path in csv_files]
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_feather("data/New_SVRM3_Ewon/combined.feather")
        return combined

combined_df = concat_csv_files()

csv_file = glob.glob("data/New_SVRM3_Ewon/20251114 SVRM3_HisLog12.csv")
random_df = [pd.read_csv(path,sep=";") for path in csv_file]

# for col in combined_df.columns:
#     if col not in random_df[0].columns:
#         print(col)

def percentage_undef_empty(df, column):
    count_undef = (df[column] == "Undef").sum()
    percentage_undef = (count_undef / len(df)) * 100

    count_empty = df[column].isna().sum()
    percentage_empty = (count_empty / len(df)) * 100

    return percentage_undef, percentage_empty

percentage_undef, percentage_empty = percentage_undef_empty(combined_df, "afgekeurd")

def clean_df(df):
    df_sorted = df.sort_values("TimeInt").reset_index(drop=True)
    df_clean = df_sorted[df_sorted["afgekeurd"] != "Undef"].reset_index(drop=True)
    return df_clean

df_clean = clean_df(combined_df)

def proto_shift_count(df, col, chunk_size):
    shift_count = 0
    previous_value = None

    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        most_frequent = chunk[col].mode()[0]  # Get the most frequent value in the chunk

        if previous_value is not None and most_frequent != previous_value:
            shift_count += 1

        previous_value = most_frequent

    print(f"Number of shifts in '{col}' with chunk size {chunk_size}: {shift_count}")
    return shift_count

proto_shift = proto_shift_count(df_clean, "afgekeurd", 100)

def shift_count_in_constant_flow(df, col, min_constant):
    shift_count = 0
    current_value = None
    current_length = 0

    for val in df[col]:
        if val == current_value:
            current_length += 1
        else:
            if current_length >= min_constant:
                shift_count += 1
            current_value = val
            current_length = 1
    print(f"Number of shifts in '{col}' with constant flow ≥ {min_constant}: {shift_count}")
    return shift_count

shift_count = shift_count_in_constant_flow(df_clean, "afgekeurd", 100)