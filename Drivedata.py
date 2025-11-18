import pandas as pd

import glob

def concat_csv_files():
    try:
        existing = pd.read_feather("data/SVRM2_Drive_data/combined.feather")
        return existing
    except:
        csv_files = glob.glob("data/SVRM2_Drive_data/*.csv")
        dfs = [pd.read_csv(path,skiprows=5) for path in csv_files]
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_feather("data/SVRM2_Drive_data/combined.feather")
        return combined

combined_df = concat_csv_files()
print(len(combined_df))

#combined = concat_csv_files()
#len(combined)

df = pd.read_csv("data/SVRM2_Drive_data/REXCURVE 2025-09-05 084850 017.CSV",skiprows=5)

print(df)