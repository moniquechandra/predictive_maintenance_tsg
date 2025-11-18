import pandas as pd

import glob

def concat_csv_files():
    csv_files = glob.glob("data/SVRM2_Drive_data/*.csv")
    dfs = [pd.read_csv(path) for path in csv_files]
    combined = pd.concat(dfs, ignore_index=True)
    return combined

combined = concat_csv_files()
len(combined)