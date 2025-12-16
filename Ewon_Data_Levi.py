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
import glob
import pandas as pd
import os

def concat_csv_files(svrm_version):
    """
    svrm_version: "svrm3" or "svrm4"
    """

    svrm_version = svrm_version.lower()
    if svrm_version not in ["svrm3", "svrm4"]:
        raise ValueError("svrm_version must be 'svrm3' or 'svrm4'")

    base_path = f"data/New_{svrm_version.upper()}_Ewon"
    feather_path = os.path.join(base_path, "combined.feather")

    try:
        existing = pd.read_feather(feather_path)
        return existing

    except FileNotFoundError:
        csv_files = glob.glob(os.path.join(base_path, "*.csv"))

        def load_csv(path):
            df = pd.read_csv(path, sep=";", dtype=str)
            print(path, df.shape)
            return df

        dfs = [load_csv(path) for path in csv_files]

        # Columns to standardize
        stangdata_cols = [
            "Pos_inloCor_1_trek13", "Pos_inloCor_1_trek24",
            "Pos_inloCor_2_trek13", "Pos_inloCor_2_trek24",
            "Pos_uitloCor_1_trek13", "Pos_uitloCor_1_trek24",
            "Pos_uitloCor_2_trek13", "Pos-UitlolCor_2_trek24"
        ]

        stanghistorie_cols = [
            "BeginVerduningPosHor", "BeginVerduningPosVert",
            "EindVerduningPosHor", "EindVerduningPosVert",
            "bgem", "dgem"
        ]

        for df in dfs:
            df.columns = [
                "StangData." + col
                if col in stangdata_cols and not col.startswith("StangData.")
                else col
                for col in df.columns
            ]

            df.columns = [
                "StangHistorie[1]." + col
                if col in stanghistorie_cols and not col.startswith("StangHistorie[1].")
                else col
                for col in df.columns
            ]

        combined = pd.concat(dfs, ignore_index=True)
        combined.to_feather(feather_path)

        return combined

def create_afgekeurd_column(df,
                            knipteller_col="knipteller",
                            goedgekeurd_col="goedgekeurd",
                            afgekeurd_col="afgekeurd"):
    df = df.copy()

    # Zet kolommen om naar numeriek, ongeldige waarden worden 0
    df[knipteller_col] = pd.to_numeric(df[knipteller_col], errors='coerce').fillna(0)
    df[goedgekeurd_col] = pd.to_numeric(df[goedgekeurd_col], errors='coerce').fillna(0)

    # Totale afgekeurd = knipteller - goedgekeurd
    df[afgekeurd_col] = (df[knipteller_col] - df[goedgekeurd_col]).clip(lower=0)

    return df

def build_feature_table(df, window_size):
    """
    Full feature engineering pipeline:
    - clean data
    - compute slopes
    - aggregate into windows
    - create future_event target

    Returns:
        agg_df (pd.DataFrame)
    """

    # -----------------------------
    # 1. CLEAN
    # -----------------------------
    df = (
        df.sort_values("TimeInt")
          .drop_duplicates(subset="TimeInt", keep="first")
          .reset_index(drop=True)
    )
    df = df[df["afgekeurd"] != "Undef"].reset_index(drop=True)

    # -----------------------------
    # 2. PREPARE NUMERIC COLUMNS
    # -----------------------------
    df["afgekeurd"] = pd.to_numeric(df["afgekeurd"], errors="coerce")
    df["afgekeurd_inc"] = df["afgekeurd"].diff().fillna(0)

    desired_cols_mapping = {
        "StangHistorie[1].BeginVerduningPosHor": ["StangHistorie[1].BeginVerduningPosHor"],
        "StangHistorie[1].BeginVerdunningPos": ["StangHistorie[1].BeginVerdunningPos"],
        "StangHistorie[1].bgem": ["StangHistorie[1].bgem", "StangHistorie[1].b_gem"],
        "StangHistorie[1].dgem": ["StangHistorie[1].dgem", "StangHistorie[1].d_gem"],
        "StangHistorie[1].EindVerduningPos": ["StangHistorie[1].EindVerduningPos"],
        "StangHistorie[1].EindVerduningPosHor": ["StangHistorie[1].EindVerduningPosHor"],
        "StangHistorie[1].L_v": ["StangHistorie[1].L_v"],
    }

    include_cols = []

    for alternatives in desired_cols_mapping.values():
        for col in alternatives:
            if col in df.columns:
                include_cols.append(col)
                break

    df["TimeInt"] = pd.to_numeric(df["TimeInt"], errors="coerce")

    df[include_cols] = df[include_cols].apply(pd.to_numeric, errors="coerce")

    # -----------------------------
    # 3. SLOPE CALCULATION
    # -----------------------------
    dt = np.diff(df["TimeInt"].values)
    dy = np.diff(df[include_cols].values, axis=0)

    slopes = pd.DataFrame(
        dy / dt[:, None],
        columns=[f"{c}_slope" for c in include_cols],
        index=df.index[1:]
    )

    slope_df = pd.concat(
        [df.loc[df.index[1:], ["TimeInt", "afgekeurd_inc"] + include_cols], slopes],
        axis=1
    )

    # -----------------------------
    # 4. WINDOW AGGREGATION
    # -----------------------------
    slope_cols = [c for c in slope_df.columns if c.endswith("_slope")]
    agg_rows = []

    for start in range(0, len(slope_df) - window_size + 1, window_size):
        window = slope_df.iloc[start:start + window_size]

        row = {
            "time_start": window["TimeInt"].iloc[0],
            "time_end": window["TimeInt"].iloc[-1],
            "sum_afgekeurd": max(window["afgekeurd_inc"].sum(), 0),
        }

        for col in slope_cols:
            base = col.replace("_slope", "")
            row[f"{base}_max_slope"] = window[col].max()
            row[f"{base}_min_slope"] = window[col].min()
            row[f"{base}_range"] = window[col].max() - window[col].min()

        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)

    # -----------------------------
    # 5. TARGET
    # -----------------------------
    agg_df["future_event"] = (
        agg_df["sum_afgekeurd"]
        .shift(-1)
        .fillna(0)
        .gt(0)
        .astype(int)
    )

    return agg_df

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

def explain_random_forest(best_rf, X_test, top_n=10):

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

def visualize_all_columns(df, output_dir="visualisations_svrm3"):
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

def plot_min_max_slopes_vs_time(df, output_dir="slope_visualizations_svrm3"):
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

def evaluate_window_sizes(df, window_sizes):
    pr_aucs = []

    for w in window_sizes:
        print(f"Evaluating window size: {w}")
        # Aggregate slopes into windows
        agg_df = build_feature_table(df, w)
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

svrm_3_data = concat_csv_files(svrm_version="svrm3")
svrm_3_df = build_feature_table(svrm_3_data, 15)
train_random_forest(svrm_3_df, use_cv=False, n_iter_search=20, use_shap=True, top_n_shap=10)
plot_min_max_slopes_vs_time(svrm_3_df)

#svrm_4_df = build_feature_table(svrm_4_data, 30)
#svrm_4_data = concat_csv_files(svrm_version="svrm4")
#svrm_4_data = create_afgekeurd_column(svrm_4_data)




# Optimal parameters found for SVRM 3 with 15 window size:
    # 'n_estimators': 500,
    # 'max_depth': 30,
    # 'min_samples_split': 10,
    # 'min_samples_leaf': 3,
    # 'max_features': 'sqrt',
    # 'class_weight': 'balanced',
    # 'random_state': 42,
    # 'n_jobs': -1

# Optimal parameters found for SVRM 4 with 30 window size:
    # 'n_estimators': 500,
    # 'max_depth': None,
    # 'min_samples_split': 10,
    # 'min_samples_leaf': 5,
    # 'max_features': 'sqrt',
    # 'class_weight': 'balanced',
    # 'random_state': 42,
    # 'n_jobs': -1