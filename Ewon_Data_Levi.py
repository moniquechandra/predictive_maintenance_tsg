import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.inspection import permutation_importance
import os
import matplotlib.pyplot as plt
import shap

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

def create_final_df(df, window_size):
    slope_cols = [col for col in df.columns if col.endswith("_slope")]
    agg_result = []

    for start in range(0, len(df) - window_size + 1, window_size):
        end = start + window_size
        window = df.iloc[start:end]

        max_slopes = window[slope_cols].max()
        min_slopes = window[slope_cols].min()
        sum_afgekeurd = window["afgekeurd_inc"].sum()

        row = {
            "time_start": window["TimeInt"].iloc[0],
            "time_end": window["TimeInt"].iloc[-1],
            "sum_afgekeurd": max(sum_afgekeurd, 0)  # already cleaned
        }

        # Add max/min slopes
        for col in slope_cols:
            base = col.replace("_slope", "")
            row[f"{base}_max_slope"] = max_slopes[col]
            row[f"{base}_min_slope"] = min_slopes[col]
            # Optional: range feature
            row[f"{base}_range"] = max_slopes[col] - min_slopes[col]

        agg_result.append(row)

    agg_df = pd.DataFrame(agg_result)

    # --- Integrate future_event target directly ---
    agg_df["future_event"] = agg_df["sum_afgekeurd"].shift(-1).fillna(0).gt(0).astype(int)

    return agg_df

agg_df = create_final_df(slope_df, 15)
print(agg_df.head())
print(agg_df["future_event"].value_counts())


#slope_df.to_excel("data/New_SVRM3_Ewon/slope_analysis.xlsx", index=False)
show_column_specs(agg_df, "sum_afgekeurd")

def plot_min_max_slopes_vs_time(df, output_dir="slope_visualizations"):
    os.makedirs(output_dir, exist_ok=True)

    # Detect base feature names
    max_cols = [c for c in df.columns if c.endswith("_max_slope")]

    for max_col in max_cols:
        base = max_col.replace("_max_slope", "")
        min_col = f"{base}_min_slope"

        if min_col not in df.columns:
            continue

        plt.figure(figsize=(12, 4))
        plt.plot(df["time_start"], df[max_col], label="Max slope")
        plt.plot(df["time_start"], df[min_col], label="Min slope")

        plt.xlabel("Time start")
        plt.ylabel("Slope")
        plt.title(f"Min / Max slope vs time — {base}")
        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, f"{base}_min_max_slope.png"))
        plt.close()

def explain_random_forest(best_rf, X_test, top_n=10):
    import shap
    import numpy as np
    import pandas as pd

    # Create SHAP explainer
    explainer = shap.TreeExplainer(best_rf)
    shap_values = explainer.shap_values(X_test)

    # For binary classification, pick class 1
    if isinstance(shap_values, list):
        shap_class1 = shap_values[1]  # shape = (n_samples, n_features)
    else:
        shap_class1 = shap_values

    # Ensure it's 2D
    if shap_class1.ndim != 2:
        shap_class1 = shap_class1.reshape(X_test.shape[0], -1)

    # Make sure the number of columns matches X_test
    shap_class1 = shap_class1[:, :X_test.shape[1]]

    # Compute mean absolute and mean SHAP values
    mean_abs_shap = np.abs(shap_class1).mean(axis=0)
    shap_mean = shap_class1.mean(axis=0)

    # Now feature names match number of columns
    feature_cols = X_test.columns[:shap_class1.shape[1]]

    # Construct DataFrame safely
    mean_shap = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': mean_abs_shap,
        'shap_mean': shap_mean
    }).sort_values(by='mean_abs_shap', ascending=False)

    top_features = mean_shap.head(top_n)
    conclusions = []

    print("\n--- Automatic Feature Conclusions ---\n")
    for idx, row in top_features.iterrows():
        feature = row['feature']
        direction = "increase" if row['shap_mean'] > 0 else "decrease"
        prob_change = abs(row['shap_mean'])
        conclusion = f"High values of '{feature}' tend to {direction} the likelihood of a future event (avg SHAP impact ≈ {prob_change:.4f})"
        conclusions.append(conclusion)
        print(conclusion)

    # Optional SHAP summary plot
    shap.summary_plot(shap_class1, X_test, plot_type="bar", show=False)

    return conclusions

def train_random_forest(df, use_cv=True, manual_params=None, n_iter_search=20, use_shap=True, top_n_shap=10):
    """
    Train a Random Forest on the given dataframe with options for hyperparameter tuning or manual parameters,
    including SHAP-based feature conclusions.
    
    Parameters:
    - df: DataFrame with features ending in "_slope" and target "future_event"
    - use_cv: bool, whether to use RandomizedSearchCV to find optimal hyperparameters
    - manual_params: dict of RandomForestClassifier parameters (used if use_cv=False)
    - n_iter_search: int, number of iterations for RandomizedSearchCV
    - use_shap: bool, whether to compute SHAP explanations
    - top_n_shap: int, number of top features to report via SHAP
    """

    feature_cols = [col for col in df.columns if col.endswith("_slope")]
    X = df[feature_cols]
    y = df["future_event"]

    # Split into training and testing
    split_index = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    if use_cv:
        # Hyperparameter distribution
        param_dist = {
            'n_estimators': [200, 500, 800, 1000],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 3, 5],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced']
        }

        # Base Random Forest
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)

        # Stratified CV for rare events
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Randomized Search CV
        rand_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=n_iter_search,
            scoring='average_precision',  # PR-AUC for rare events
            cv=cv,
            n_jobs=-1,
            random_state=42,
            verbose=2
        )

        # Fit Randomized Search
        rand_search.fit(X_train, y_train)
        print("Best hyperparameters:", rand_search.best_params_)
        print("Best PR-AUC (CV):", rand_search.best_score_)

        best_rf = rand_search.best_estimator_

    else:
        # Use manual parameters
        if manual_params is None:
            manual_params = {
                'n_estimators': 500,
                'max_depth': 30,
                'min_samples_split': 10,
                'min_samples_leaf': 3,
                'max_features': 'sqrt',
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }
        best_rf = RandomForestClassifier(**manual_params)
        best_rf.fit(X_train, y_train)
        print("Random Forest trained with manual parameters:", manual_params)

    # Permutation importance
    perm = permutation_importance(best_rf, X_test, y_test, n_repeats=20, random_state=42, n_jobs=-1)
    perm_importance = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
    print("\nTop 15 feature importances (permutation importance):")
    print(perm_importance.head(15))

    # PR-AUC / ROC-AUC evaluation
    y_proba = best_rf.predict_proba(X_test)[:,1]
    print("\nModel Evaluation:")
    print("PR-AUC :", average_precision_score(y_test, y_proba))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    # SHAP explanations
    if use_shap:
        print("\nGenerating SHAP-based feature conclusions...")
        top_conclusions = explain_random_forest(best_rf, X_test, top_n=top_n_shap)
    else:
        top_conclusions = []

    return best_rf, top_conclusions

train_random_forest(agg_df, use_cv=False, n_iter_search=20, use_shap=True, top_n_shap=10)

def evaluate_window_sizes(slope_df, window_sizes):
    pr_aucs = []

    for w in window_sizes:
        print(f"Evaluating window size: {w}")
        # Aggregate slopes into windows
        agg_df = create_final_df(slope_df, w)
        agg_df.loc[agg_df["sum_afgekeurd"] <= 0, "sum_afgekeurd"] = 0
        agg_df["future_event"] = agg_df["sum_afgekeurd"].shift(-1).fillna(0).gt(0).astype(int)

        feature_cols = [col for col in agg_df.columns if col.endswith("_slope")]
        X = agg_df[feature_cols]
        y = agg_df["future_event"]

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train a small Random Forest (faster) for evaluation
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)

        # Predict probabilities for PR-AUC
        y_proba = clf.predict_proba(X_test)[:,1]
        pr_auc = average_precision_score(y_test, y_proba)
        pr_aucs.append(pr_auc)

    # Plot PR-AUC vs window size
    plt.figure(figsize=(10,5))
    plt.plot(window_sizes, pr_aucs, marker='o')
    plt.xlabel("Window size")
    plt.ylabel("PR-AUC")
    plt.title("PR-AUC vs Window Size")
    plt.grid(True)
    plt.show()

#evaluate_window_sizes(slope_df, window_sizes=[10, 15, 20, 25, 30, 40, 50], n_iter_search=5)