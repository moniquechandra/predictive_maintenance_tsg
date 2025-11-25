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

count_undef = (combined_df["afgekeurd"] == "Undef").sum()
percentage_undef = (count_undef / len(combined_df)) * 100

count_empty = combined_df["afgekeurd"].isna().sum()
percentage_empty = (count_empty / len(combined_df)) * 100

print(f"Percentage of 'Undef' in 'afgekeurd' column: {percentage_undef:.2f}%")
print(f"Percentage of empty/NaN values in 'afgekeurd' column: {percentage_empty:.2f}%")

df_sorted = combined_df.sort_values("TimeInt").reset_index(drop=True)
df_clean = df_sorted[df_sorted["afgekeurd"] != "Undef"].reset_index(drop=True)
print(df_clean)

chunk_size = 100
previous_counts = None
change_count = 0

for i in range(0, len(df_clean), chunk_size):
    chunk = df_clean.iloc[i:i+chunk_size]
    counts = chunk["afgekeurd"].value_counts(dropna=False)  # include NaN if any
    #print(f"Rows {i} to {i+chunk_size-1}:")
    #print(counts)
    
    if previous_counts is not None:
        if not counts.equals(previous_counts):
            print(" -> Counts changed from previous chunk!\n")
            change_count += 1
        else:
            print(" -> Counts are the same as previous chunk.\n")
    
    previous_counts = counts

print(change_count)