import pandas as pd
import numpy as np

def clean_data(df):
    """
    Cleans and preprocesses time-series sensor data.

    This function performs the following steps:
    - Removes fully duplicated rows
    - Converts the 'TimeStr' column to datetime
    - Casts feature columns (from the third column onward) to numeric values
    - Standardizes missing values ('Undef', 'nan' -> NaN)
    - Removes rows where all feature values are missing
    - Drops columns that contain only missing values
    - Resolves duplicate timestamps by keeping the row with the fewest missing values
    - Removes currently irrelevant 'wisselblok_settings' columns
    - Sorts the data chronologically by timestamp

    Parameters
    ----------
    df : pandas.DataFrame
        Raw input data containing a 'TimeStr' column and sensor features.

    Returns
    -------
    pandas.DataFrame
        Cleaned and chronologically ordered DataFrame ready for analysis.
    """
    df = df.drop_duplicates()
    
    df = df.copy()
    df['TimeStr'] = pd.to_datetime(df['TimeStr'], format="%d/%m/%Y %H:%M:%S")
    
    df.iloc[:, 2:] = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
    
    df = df.replace(["Undef", "nan"], np.nan)
    
    mask = (df.iloc[:, 2:].isna()).all(axis=1)
    print(len(mask), "rows only have NaN entries (except Timestamps)")
    df_not_nan = df[~mask]
    
    df_not_nan = df_not_nan.dropna(axis=1, how='all')
    
    subset=["TimeStr"]
    
    print(len(df_not_nan[df_not_nan.duplicated(subset=subset)]), "rows are duplicated on TimeStr")
    df_cleaned = df_not_nan.loc[
    df_not_nan.assign(n_missing=df_not_nan.isna().sum(axis=1))
      .sort_values("n_missing")
      .drop_duplicates(subset=subset, keep="first")
      .index
]
    
    # temporary: remove wisselblok settings columns, none of them is relevant currently
    cols_to_drop = [col for col in df_cleaned.columns if "wisselblok_settings" in col.lower()]
    df_cleaned = df_cleaned.drop(columns=cols_to_drop)

    df_cleaned = df_cleaned.sort_values('TimeStr')
    
    print(len(df_cleaned), "rows retained. Data is cleaned")
    return df_cleaned

def combine_columns(df, col1, col2, new_col_name=None):
    """
    Combines two columns into one.
    Priority: col1 value first, otherwise col2 value.
    If both are NaN → result is NaN.

    Parameters:
        df (pd.DataFrame): the original dataframe
        col1 (str): first column name
        col2 (str): second column name
        new_col_name (str): name for the new column 
                            (default: col1)

    Returns:
        pd.DataFrame with the combined column
    """

    if new_col_name is None:
        new_col_name = col1

    df[new_col_name] = df[col1].combine_first(df[col2])
    df = df.drop([col2], axis=1)
    return df

# SVRM3 EWON TOTAAL AANTAL ONLY CONTAINS UNDEF AND 0