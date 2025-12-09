import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt

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
        # Standardize column names by prefixing with "StangData." and "StangHistorie[1]." if not already present
        for df in dfs:
            df.columns = ["StangData." + col if not col.startswith("StangData.") and col in ["Pos_inloCor_1_trek13", "Pos_inloCor_1_trek24", "Pos_inloCor_2_trek13", "Pos_inloCor_2_trek24", "Pos_uitloCor_1_trek13", "Pos_uitloCor_1_trek24", "Pos_uitloCor_2_trek13", "Pos-UitlolCor_2_trek24"] else col for col in df.columns]
            df.columns = ["StangHistorie[1]." + col if col in ["BeginVerduningPosHor", "BeginVerduningPosVert", "EindVerduningPosHor", "EindVerduningPosVert","bgem","dgem"] else col for col in df.columns]
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
    df_sorted = df.sort_values("TimeInt").drop_duplicates(subset="TimeInt",keep="first").reset_index(drop=True)
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

#proto_shift = proto_shift_count(df_clean, "afgekeurd", 100)

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

#shift_count = shift_count_in_constant_flow(df_clean, "afgekeurd", 100)

def visualize_all_columns(df, output_dir="new_visualisations"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Remove columns with only "Undef" values
    cols_to_plot = [col for col in df.columns if not (df[col] == "Undef").all()]
    
    for col in cols_to_plot:
        try:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(pd.to_numeric(df[col], errors='coerce'))
            ax.set_title(col)
            ax.set_xlabel("Index")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{col}.png"))
            plt.close()
        except:
            print(f"Cannot plot {col}")

#visualize_all_columns(df_clean)

def show_column_specs(df, column_name):
    if column_name in df.columns:
        unique_values = df[column_name].unique()
        num_unique = len(unique_values)
        data_type = df[column_name].dtype
        print(f"Column: {column_name}")
        print(f"Data Type: {data_type}")
        print(f"Number of Unique Values: {num_unique}")
        print(f"Unique Values: {unique_values}")
    else:
        print(f"Column '{column_name}' does not exist in the DataFrame.")

#show_column_specs(df_clean, "afgekeurd")


def calculate_max_min_slope(df):
    df["afgekeurd"] = pd.to_numeric(df["afgekeurd"], errors="coerce")
    df["afgekeurd_inc"] = df["afgekeurd"].diff().fillna(0)
    include_cols_maybe = ["d_hor_offset","d_vert_offset,","StangData.Correctie_Totale_lengte","StangData.Correctie_Totale_lengte_operator_trek1en3",
                    "StangData.Offset_knipklem_knippositie_Operator","StangData.Pos_inloCor_1_trek13","StangData.Pos_inloCor_1_trek24",
                    "StangData.Pos_inloCor_2_trek13","StangData.Pos_inloCor_2_trek24","StangData.Pos_uitloCor_1_trek13", "StangData.Pos_uitloCor_1_trek24",
                    "StangData.Pos_uitloCor_2_trek13","StangData.Pos-UitlolCor_2_trek24","VU-Center_hor","Wisselblok_settings[2].Wp1_diameter_offset",
                    "Wisselblok_settings[2].Wp2_diameter_offset"]
    include_cols = ["StangHistorie[1].BeginVerduningPosHor","StangHistorie[1].BeginVerdunningPos","StangHistorie[1].bgem","StangHistorie[1].dgem",
                    "StangHistorie[1].EindVerduningPos","StangHistorie[1].EindVerduningPosHor","StangHistorie[1].L_v"]
    
    df["TimeInt"] = df["TimeInt"].apply(pd.to_numeric, errors="coerce")
    df[include_cols] = df[include_cols].apply(pd.to_numeric, errors="coerce")

    dt = np.diff(df["TimeInt"].values)
    dy = np.diff(df[include_cols].values, axis=0)

    slopes = pd.DataFrame(
        dy / dt[:, None],
        columns=[f"{c}_slope" for c in include_cols],
        index=df.index[1:]
)
    result = pd.concat([df[["TimeInt","afgekeurd_inc"] + include_cols],slopes], axis=1)

    return result

slope_df = calculate_max_min_slope(df_clean)

print(slope_df.columns)

def create_final_df(df,window_size):
    slope_cols = [col for col in df.columns if col.endswith("_slope")]
    agg_result = []
    for start in range(0, len(df) - window_size + 1, window_size):
        end = start + window_size
        window = df.iloc[start:end]
        max_slopes = window.loc[:, slope_cols].max()
        sum_afgekeurd = window["afgekeurd_inc"].sum()
        
        # Optional: include group/time of first row
        row = {
            "time_start": window["TimeInt"].iloc[0],
            "time_end": window["TimeInt"].iloc[-1],
            **max_slopes.to_dict(),
            "sum_afgekeurd": sum_afgekeurd
        }
        agg_result.append(row)

    agg_df = pd.DataFrame(agg_result)
    return agg_df

agg_df = create_final_df(slope_df,100)
agg_df.loc[agg_df["sum_afgekeurd"] <= 0, "sum_afgekeurd"] = 0

#slope_df.to_excel("data/New_SVRM3_Ewon/slope_analysis.xlsx", index=False)

print(agg_df)

show_column_specs(agg_df, "sum_afgekeurd")

plt.figure()
plt.plot(agg_df["time_start"], agg_df["sum_afgekeurd"])
plt.xlabel("Time")
plt.ylabel("Extra sum")
plt.title("Extra sum per window")
plt.show()


def train_random_forest(df):
    feature_cols = [col for col in df.columns if col.endswith("_slope")]
    X = df[feature_cols]
    y = df["sum_afgekeurd"]

    # Encode target variable
    y_encoded = LabelEncoder().fit_transform(y)

    # Split data into training and testing sets
    split_index = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y_encoded[:split_index], y_encoded[split_index:]

    # Train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Evaluate the model
    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)

    return clf

# --- configuration ---
time_col = "time_start"
output_dir = "final_visualisations"

# create folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# ensure time is numeric
agg_df[time_col] = pd.to_numeric(agg_df[time_col], errors="coerce")

# select columns to plot (exclude time)
plot_cols = [
    col for col in agg_df.columns
    if col != time_col and pd.api.types.is_numeric_dtype(agg_df[col])
]

# --- plotting loop ---
for col in plot_cols:
    plt.figure()
    plt.plot(agg_df[time_col], agg_df[col])
    plt.xlabel("Time")
    plt.ylabel(col)
    plt.title(f"{col} vs Time")
    plt.tight_layout()

    filename = os.path.join(output_dir, f"{col}.png")
    plt.savefig(filename, dpi=150)
    plt.close()  # important: frees memory

#train_random_forest(agg_df)