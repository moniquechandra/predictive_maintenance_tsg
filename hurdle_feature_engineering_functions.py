import pandas as pd
import numpy as np

def add_increment_column(df, column_name, decimal_places=6, reset_threshold=1e-6):
    """
    Computes per-row increments for a cumulative or counter-like column.

    This function creates a new column that represents the change between
    consecutive rows of a specified numeric column. It is designed for
    cumulative counters and handles floating-point noise and true counter
    resets gracefully.

    The function:
    - Converts the target column to numeric (non-numeric values become NaN)
    - Computes differences between consecutive rows
    - Treats very small changes as zero to avoid floating-point artifacts
    - Detects true counter resets and replaces negative jumps with the
      current value
    - Stores the result in a new column named '<column_name>_inc'

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the cumulative column.
    column_name : str
        Name of the column for which increments are computed.
    decimal_places : int, optional
        Number of decimal places to round values to before differencing
        (default is 6).
    reset_threshold : float, optional
        Threshold used to distinguish floating-point noise from true counter
        resets (default is 1e-6).

    Returns
    -------
    pandas.DataFrame
        DataFrame with an additional '<column_name>_inc' column containing
        the computed increments.
    """
    series = pd.to_numeric(df[column_name], errors="coerce")

    inc_col = f"{column_name}_inc"
    df[inc_col] = np.nan

    prev = series.shift(1)
    valid_mask = series.notna() & prev.notna()

    current = series[valid_mask].round(decimal_places)
    previous = prev[valid_mask].round(decimal_places)

    increments = current - previous

    # Force near-zero values to zero
    increments[np.abs(increments) < reset_threshold] = 0.0

    # Handle true counter resets only
    reset_mask = increments < -reset_threshold
    increments[reset_mask] = current[reset_mask]

    df.loc[valid_mask, inc_col] = increments

    return df

def compute_raw_slopes(df, timestamp_col="TimeInt", sensor_pattern="StangHistorie"):
    """
    Compute instantaneous slopes on raw high-frequency sensor data.
    This should be the first step of the feature engineering.

    This function calculates first-order numerical derivatives
    (difference quotient) for each sensor signal based on consecutive
    timestamp differences. It should be applied to *raw*, unaggregated
    data before any window-based aggregation.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe with high-frequency measurements.
    timestamp_col : str, default="TimeInt"
        Name of the timestamp column. Must be numeric and monotonically increasing.
    sensor_pattern : str, default="StangHistorie"
        Substring used to identify sensor columns for slope computation.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns containing raw slopes for each
        identified sensor column. New columns are suffixed with `_raw_slope`.

    Notes
    -----
    - Slopes are computed as `diff(sensor) / diff(time)`.
    """

    df = df.sort_values(timestamp_col).copy()

    sensor_cols = df.columns[df.columns.str.contains(sensor_pattern)].tolist()

    dt = df[timestamp_col].diff()
    
    for col in sensor_cols:
        df[f"{col}_raw_slope"] = df[col].diff() / dt

    return df

def aggregate_raw_data(df, agg_seconds, timestamp_col="TimeInt", sensor_pattern="StangHistorie"):
    """
    Aggregate raw high-frequency data into fixed time windows.
    This should be the step after aggregating raw slopes.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe with high-frequency measurements.
    agg_seconds : int
        Aggregation window in seconds.
    timestamp_col : str
        Name of the timestamp column.
    sensor_pattern : str, default="StangHistorie"
        Substring used to identify sensor columns for aggregation

    Returns
    -------
    pd.DataFrame
        Aggregated dataframe with summary statistics.
    """

    print(f"Aggregating raw data with {agg_seconds}s windows...")
    print(f"Original shape: {df.shape}")

    df = df.sort_values(timestamp_col).copy()
    df["agg_bin"] = (df[timestamp_col] // agg_seconds).astype(int)

    # Identify sensor columns
    param_cols = df.columns[df.columns.str.contains(sensor_pattern)].tolist()

    # Build aggregation dictionary
    agg_dict = {col: ["mean", "std"] for col in param_cols}

    # Sum rejection counters
    for col in ["afgekeurd_inc", "goedgekeurd_inc"]:
        if col in df.columns:
            agg_dict[col] = "sum"

    # Keep earliest timestamp per window
    agg_dict[timestamp_col] = "first"

    df_agg = df.groupby("agg_bin").agg(agg_dict)
    df_agg.columns = [f"{c}_{s}" for c, s in df_agg.columns]
    df_agg = df_agg.reset_index(drop=True)

    print(f"Aggregated shape: {df_agg.shape}")
    return df_agg

def aggregate_raw_slope_features(df,
                                 agg_seconds,
                                 timestamp_col="TimeInt",
                                 sensor_pattern="StangHistorie"):
    """
    Aggregate extreme raw slope behavior within fixed time windows.

    This function summarizes instantaneous sensor slopes (previously
    computed on raw data) by aggregating their extreme and variability
    characteristics within each aggregation window. It is intended to be
    applied after raw slope computation and before higher-level
    feature engineering.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing raw slope columns (suffixed with `_raw_slope`).
    agg_seconds : int
        Aggregation window size in seconds.
    timestamp_col : str, default="TimeInt"
        Name of the timestamp column used to assign aggregation bins.
    sensor_pattern : str, default="StangHistorie"
        Substring used to identify sensor-related slope columns.

    Returns
    -------
    pd.DataFrame
        Aggregated dataframe containing summary statistics of raw slopes
        per time window. Output columns are suffixed with the applied
        aggregation statistic (e.g. `_max`, `_min`, `_std`).
    """

    df = df.copy()
    df["agg_bin"] = (df[timestamp_col] // agg_seconds).astype(int)

    slope_cols = [
        c for c in df.columns
        if c.endswith("_raw_slope") and sensor_pattern in c
    ]

    agg_dict = {}

    for col in slope_cols:
        base = col.replace("_raw_slope", "")
        agg_dict[col] = ["max", "min", "std"]

    df_slope_agg = df.groupby("agg_bin").agg(agg_dict)
    df_slope_agg.columns = [
        f"{c}_{s}" for c, s in df_slope_agg.columns
    ]

    df_slope_agg = df_slope_agg.reset_index(drop=True)

    return df_slope_agg

def add_volatility_features_aggregated(df, window_sizes=[4]):
    """
    Add volatility features to aggregated data.
    Uses rolling windows over aggregated time windows.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated dataframe with slope features.
    window_sizes : list of int
        Rolling window sizes in number of aggregated windows.

    Returns
    -------
    pd.DataFrame
        DataFrame with volatility features added.
    """

    print(f"Adding volatility features with window sizes: {window_sizes}...")

    df = df.copy()
    param_cols = [c for c in df.columns if c.endswith("_mean")]

    for col in param_cols:
        base = col.replace("_mean", "")

        for w in window_sizes:
            roll = df[col].rolling(w)

            df[f"{base}_std_{w}w"] = roll.std().shift(1)
            df[f"{base}_range_{w}w"] = (roll.max() - roll.min()).shift(1)
            df[f"{base}_min_{w}w"] = roll.min().shift(1)
            df[f"{base}_max_{w}w"] = roll.max().shift(1)
            df[f"{base}_mean_{w}w"] = roll.mean().shift(1)

    return df

def create_full_feature_pipeline(df,
                                 agg_seconds=30,
                                 volatility_windows=[2],
                                 timestamp_col="TimeInt"):
    """
    Complete pipeline: Aggregate raw data → Engineer features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw high-frequency dataframe.
    agg_seconds : int
        Aggregation window in seconds.
    volatility_windows : list of int
        Window sizes for volatility features.
    timestamp_col : str
        Name of timestamp column.

    Returns
    -------
    pd.DataFrame
        Fully processed dataframe with all features.
    """
    
    df_raw_slopes = compute_raw_slopes(df, timestamp_col)
    df_agg_levels = aggregate_raw_data(df_raw_slopes, agg_seconds=agg_seconds, timestamp_col=f"{timestamp_col}")
    df_agg_slopes = aggregate_raw_slope_features(df_raw_slopes, agg_seconds=agg_seconds, timestamp_col=f"{timestamp_col}")
    df_agg = pd.concat([df_agg_levels, df_agg_slopes], axis=1)
    df_agg = df_agg.loc[:, ~df_agg.columns.duplicated()]

    df_final = add_volatility_features_aggregated(
        df_agg,
        window_sizes=volatility_windows
    )

    print(f"Number of features added: {len(df_final.columns)}")

    return df_final