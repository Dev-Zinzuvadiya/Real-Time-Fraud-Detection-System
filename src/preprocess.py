import numpy as np
import pandas as pd

# FILE PATHS
INPUT_FILE = "data/raw/fraud_transactions.csv"
OUTPUT_FILE = "data/processed/fraud_processed.csv"
REPORT_FILE = "fraud_data_quality_report.txt"


# LOAD DATA
print(">> LOADING DATASET...")
df = pd.read_csv(INPUT_FILE)

print(">> DATASET LOADED SUCCESSFULLY!!")
print(f"\n:> ORIGINAL SHAPE: {df.shape}")


# BASIC INFORMATION
original_rows = len(df)
original_columns = len(df.columns)
missing_before = df.isnull().sum().sum()
duplicates_before = df.duplicated().sum()


# REMOVE DUPLICATE ROWS
print("\n\n>> CHECKING DUPLICATES ROWs ↘")
print("-" * 30)
df = df.drop_duplicates()

duplicates_removed = original_rows - len(df)
print(f":> DUPLICATE ROWs REMOVED: {duplicates_removed}")


# HANDLE MISSING NUMERICAL VALUES
print("\n\n>> HANDLING NUMERICAL MISSING VALUEs ↘")
print("-" * 30)
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

for column in numeric_columns:
    if column == "is_fraud":
        continue

    missing_count = df[column].isnull().sum()

    if missing_count > 0:
        median_value = df[column].median()
        df[column] = df[column].fillna(median_value)
        print(f":> {column}: {missing_count} VALUE FILLED USING 'MEDIAN'...")


# HANDLE MISSING CATEGORICAL VALUES
print("\n\n>> HANDLING CATEGORICAL MISSING VALUEs ↘")
print("-" * 30)
categorical_columns = df.select_dtypes(include="object").columns.tolist()

for column in categorical_columns:
    missing_count = df[column].isnull().sum()

    if missing_count > 0:
        mode_value = df[column].mode()[0]
        df[column] = df[column].fillna(mode_value)
        print(
            f":> {column}: {missing_count} VALUE FILLED USING 'MODE (most-frequent)'..."
        )


# CHECK REMAINING MISSING VALUES
missing_after = df.isnull().sum().sum()
print(f"\n:> REMAINING MISSING VALUEs: {missing_after}")


# CHECK TARGET
if "is_fraud" not in df.columns:
    raise ValueError("\n>> TARGET COLUMN 'is_fraud' WAS NOT FOUND...")


# REMOVE UNECCESSARY COLUMNs
remove_columns = [
    "home_country",
    "customer_transaction_count_30d",
    "transaction_timestamp",
    "new_merchant_flag",
    "home_city",
    "avg_transaction_amount_30d",
    "browser",
    "new_device_flag",
    "currency",
    "time_since_last_transaction_minutes",
    "distance_from_home_km",
    "transactions_last_1h",
    "merchant_risk_score",
    "merchant_transaction_volume",
    "customer_income",
    "operating_system",
    "customer_segment",
]


# SEPARATE FEATURES AND TARGET
X = df.drop(columns=remove_columns)
y = df["is_fraud"]

processed_df = X

print(":> FEATURES:", X.shape)
print(":> TARGET:", y.shape)

processed_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n:> PROCESSED DATASET SAVED TO: {OUTPUT_FILE}")


# CREATE DATA QUALITY REPORT
with open(REPORT_FILE, "w", encoding="utf-8") as report:
    report.write("FRAUD DATA QUALITY REPORT\n")
    report.write("=========================\n")
    report.write(f"- Original rows: {original_rows}\n")
    report.write(f"- Original columns: {original_columns}\n")
    report.write(f"- Missing values before cleaning: " f"{missing_before}\n")
    report.write(f"- Duplicate rows before cleaning: " f"{duplicates_before}\n")
    report.write(f"- Duplicate rows removed: " f"{duplicates_removed}\n")
    report.write(f"- Remaining missing values: " f"{missing_after}\n")
    report.write(f"\n- Final rows: {len(processed_df)}\n")
    report.write(f"- Final columns: {len(processed_df.columns)}\n\n")

    report.write("TARGET DISTRIBUTION\n")
    report.write("-------------------\n")
    report.write(str(df["is_fraud"].value_counts()))

    report.write("\n\nTARGET PERCENTAGE\n")
    report.write("-----------------\n")
    report.write(str(df["is_fraud"].value_counts(normalize=True) * 100))

print("\n>> DATA PRE-PROCESSING COMPLETED SUCCESSFULLY!!")
