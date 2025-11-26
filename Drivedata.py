import pandas as pd
import numpy as np
import glob

def concat_csv_files():
    try:

        existing = pd.read_feather("data/SVRM2_Drive_data/combined.feather")
        return existing
    
    except:

        first_dataframe = glob.glob("data/SVRM2_Drive_data/REXCURVE 2025-09-05 084850 017.csv")
        dfs_headers = [pd.read_csv(first_dataframe[0],skiprows=1,nrows=2)]
        new_column_names = []
        for col in dfs_headers[0].columns:
            new_column_name = str(dfs_headers[0][col].iloc[0]) + "_" + str(dfs_headers[0][col].iloc[1])
            new_column_names.append(new_column_name)
        
        csv_files = glob.glob("data/SVRM2_Drive_data/*.csv")
        dfs = [pd.read_csv(path,skiprows=5) for path in csv_files]
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.iloc[:, :84]
        combined.columns = new_column_names

        combined.to_feather("data/SVRM2_Drive_data/combined.feather")
        return combined

combined_df = concat_csv_files()

def create_diffs_df(df,keys):
    diff_lists = {}

    for key in keys:
        diff_lists[key] = np.abs(
            df[f"{key}_Actual position value"] -
            df[f"{key}_Target position"]
        )
    diff_df = pd.DataFrame(diff_lists)
    return diff_df

keys = ["22", "24", "26", "28", "30"]

diff_df = create_diffs_df(combined_df, keys)

def create_top100_df(df):
    top100_df = pd.DataFrame({
        key: diff_df[key].nlargest(100).values
        for key in df.columns
    })
    top100_df.to_excel("data/SVRM2_Drive_data/top100_diffs.xlsx", index=False)
    return top100_df

top100_df = create_top100_df(diff_df)

thresholds = [200, 100, 50, 10, 5, 2, 1]

result = pd.DataFrame(
    {
        key: [(diff_df[key] > t).sum() for t in thresholds]
        for key in diff_df.columns
    },
    index=thresholds
)

result.to_excel("data/SVRM2_Drive_data/diff_thresholds_counts.xlsx")

print(result)