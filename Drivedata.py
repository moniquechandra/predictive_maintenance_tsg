import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

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
print(len(combined_df))
print(combined_df)

combined_df = combined_df[:10000]

corr = combined_df.corr().abs()

pairs = []
columns = corr.columns

for i, j in itertools.combinations(columns, 2):
    pairs.append((i, j, corr.loc[i, j]))

corr_pairs = pd.DataFrame(pairs, columns=["feature_1", "feature_2", "corr_value"])

least_corr_pairs = corr_pairs.sort_values("corr_value").head(50)

print(least_corr_pairs)