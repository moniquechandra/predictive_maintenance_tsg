import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


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

def percentage_undef_empty(df, column):
    count_undef = (df[column] == "Undef").sum()
    percentage_undef = (count_undef / len(df)) * 100

    count_empty = df[column].isna().sum()
    percentage_empty = (count_empty / len(df)) * 100

    return percentage_undef, percentage_empty

percentage_undef, percentage_empty = percentage_undef_empty(combined_df, "afgekeurd")

def clean_df(df):
    df_sorted = df.sort_values("TimeInt").reset_index(drop=True)
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

proto_shift = proto_shift_count(df_clean, "afgekeurd", 100)

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

shift_count = shift_count_in_constant_flow(df_clean, "afgekeurd", 100)

df_clean = df_clean[pd.to_numeric(df_clean["afgekeurd"], errors='coerce').notna()].reset_index(drop=True)
df_clean["afgekeurd"] = pd.to_numeric(df_clean["afgekeurd"], errors='coerce')
df_clean["afgekeurd_next"] = df_clean["afgekeurd"].shift(-1)
df_clean["will_shift"] = (df_clean["afgekeurd"] != df_clean["afgekeurd_next"]).astype(int)


train_size = int(len(df_clean) * 0.8)
train = df_clean.iloc[:train_size]
test = df_clean.iloc[train_size:]

X_train = train.drop(columns=["afgekeurd", "afgekeurd_next", "will_shift", "TimeInt", "TimeStr"])
y_train = train["will_shift"]

X_test = test.drop(columns=["afgekeurd", "afgekeurd_next", "will_shift", "TimeInt", "TimeStr"])
y_test = test["will_shift"]

cat_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

for col in cat_cols:
    X_train[col] = encoder.fit_transform(X_train[[col]])
    X_test[col] = encoder.transform(X_test[[col]])

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)
pred = model.predict(X_test)
print(classification_report(y_test, pred))

fi = pd.Series(model.feature_importances_, index=X_train.columns)
print("Top predictive features:\n", fi.sort_values(ascending=False).head(20))