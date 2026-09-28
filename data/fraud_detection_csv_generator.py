import os
import math
import numpy as np
import pandas as pd

# CONFIGURATION
ROW_COUNT = 500_000
CHUNK_SIZE = 50_000
OUTPUT_FILE = "fraud_transactions.csv"
REPORT_FILE = "fraud_data_quality_report.txt"

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)


# >>> NEW: Set Ratio (0.5 = balanced) or (0.03 = Imbalanced Realistic).
TARGET_RATE = 0.50

# REFERENCE VALUEs
GENDERS = np.array(["Male", "Female", "Other"])

COUNTRIES = np.array(
    [
        "India",
        "United States",
        "United Kingdom",
        "Canada",
        "Australia",
        "Germany",
        "France",
        "Singapore",
        "UAE",
        "South Africa",
    ]
)

CITIES = np.array(
    [
        "Mumbai",
        "Delhi",
        "Surat",
        "Ahmedabad",
        "Bangalore",
        "London",
        "New York",
        "Toronto",
        "Sydney",
        "Dubai",
        "Berlin",
        "Paris",
    ]
)

CUSTOMER_SEGMENTS = np.array(["Standard", "Premium", "VIP", "Business"])

PAYMENT_METHODS = np.array(
    ["Credit Card", "Debit Card", "UPI", "Net Banking", "Digital Wallet"]
)

MERCHANT_CATEGORIES = np.array(
    [
        "Electronics",
        "Grocery",
        "Travel",
        "Fashion",
        "Restaurant",
        "Gaming",
        "Entertainment",
        "Healthcare",
        "Education",
        "Online Services",
    ]
)

DEVICE_TYPES = np.array(["Mobile", "Desktop", "Tablet"])
BROWSERS = np.array(["Chrome", "Firefox", "Safari", "Edge", "Opera"])
OPERATING_SYSTEMS = np.array(["Windows", "Android", "iOS", "macOS", "Linux"])


# HELPER FUNCTIONs
def random_choice(values, size, probabilities=None):
    return rng.choice(values, size=size, p=probabilities)


def inject_missing_values(df, column, percentage):
    count = int(len(df) * percentage)
    if count <= 0:
        return

    indices = rng.choice(len(df), size=count, replace=False)
    df.loc[indices, column] = np.nan


def inject_outliers(df, column, percentage, multiplier=40):
    count = int(len(df) * percentage)
    if count <= 0:
        return

    indices = rng.choice(len(df), size=count, replace=False)
    df.loc[indices, column] = df.loc[indices, column].astype(float) * multiplier


def inject_categorical_inconsistency(df, column, percentage):
    count = int(len(df) * percentage)
    if count <= 0:
        return

    indices = rng.choice(len(df), size=count, replace=False)

    variants = {
        "Credit Card": ["credit card", "CreditCard", " Credit Card ", "CREDIT CARD"],
        "Debit Card": ["debit card", "DebitCard", " debit card "],
        "UPI": ["upi", " UPI ", "Upi"],
        "Mobile": ["mobile", " MOBILE", "Mobile "],
        "Desktop": ["desktop", " DESKTOP"],
        "India": ["india", " INDIA", " India "],
        "United States": ["USA", "US", "united states", " United States "],
    }

    for index in indices:
        original = df.at[index, column]

        if pd.isna(original):
            continue
        if original in variants:
            df.at[index, column] = rng.choice(variants[original])


def balance_labels(fraud_probabilities, target_rate=TARGET_RATE):
    """
    Assign is_fraud=1 to the top `target_rate` fraction of fraud_probability.
    This preserves the original ranking (so patterns stay realistic) while
    forcing a desired class balance.
    """
    n = len(fraud_probabilities)
    k = int(round(n * target_rate))
    labels = np.zeros(n, dtype=np.int8)

    if k <= 0:
        return labels

    # top-k indices by probability
    top_idx = np.argpartition(fraud_probabilities, -k)[-k:]
    labels[top_idx] = 1
    return labels


# GENERATE ONE CHUNK
def generate_chunk(start_index, rows):
    customer_ids = np.array(
        [
            f"CUST_{x:08d}"
            for x in rng.integers(1, max(10_000, ROW_COUNT // 3), size=rows)
        ]
    )

    transaction_ids = np.array([f"TXN_{start_index + x:010d}" for x in range(rows)])

    merchant_ids = np.array(
        [f"MER_{x:06d}" for x in rng.integers(1, 50_000, size=rows)]
    )

    device_ids = np.array([f"DEV_{x:07d}" for x in rng.integers(1, 500_000, size=rows)])

    # CUSTOMER INFORMATION
    age = rng.integers(18, 75, size=rows)
    gender = random_choice(GENDERS, rows, probabilities=[0.48, 0.49, 0.03])

    country = random_choice(
        COUNTRIES,
        rows,
        probabilities=[0.40, 0.15, 0.08, 0.06, 0.05, 0.06, 0.05, 0.04, 0.06, 0.05],
    )

    city = random_choice(CITIES, rows)

    customer_segment = random_choice(
        CUSTOMER_SEGMENTS, rows, probabilities=[0.65, 0.25, 0.07, 0.03]
    )

    account_age_days = rng.integers(30, 3650, size=rows)

    customer_income = np.round(
        rng.lognormal(mean=np.log(60_000), sigma=0.55, size=rows)
    )

    # TRANSACTION INFORMATION
    transaction_amount = np.around(
        rng.lognormal(mean=np.log(120), sigma=1.0, size=rows), 2
    )

    payment_method = random_choice(
        PAYMENT_METHODS, rows, probabilities=[0.25, 0.20, 0.30, 0.10, 0.15]
    )

    merchant_category = random_choice(MERCHANT_CATEGORIES, rows)
    device_type = random_choice(DEVICE_TYPES, rows, probabilities=[0.65, 0.30, 0.05])
    browser = random_choice(BROWSERS, rows)
    operating_system = random_choice(OPERATING_SYSTEMS, rows)

    # TIME
    start_date = pd.Timestamp("2023-01-01")
    random_days = rng.integers(0, 1095, size=rows)
    random_seconds = rng.integers(0, 86_400, size=rows)

    transaction_timestamp = (
        start_date
        + pd.to_timedelta(random_days, unit="D")
        + pd.to_timedelta(random_seconds, unit="s")
    )

    timestamp_str = transaction_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    hour = transaction_timestamp.hour.to_numpy()

    # CUSTOMER BEHAVIOR
    transactions_last_1h = rng.poisson(1.2, size=rows)
    transactions_last_24h = rng.poisson(5, size=rows)
    failed_transactions_24h = rng.poisson(0.5, size=rows)

    avg_transaction_amount_30d = np.around(
        rng.lognormal(np.log(100), 0.65, size=rows), 2
    )

    customer_transaction_count_30d = rng.poisson(12, size=rows)
    new_device_flag = rng.binomial(1, 0.12, size=rows)
    new_merchant_flag = rng.binomial(1, 0.15, size=rows)
    distance_from_home_km = np.round(rng.exponential(20, size=rows), 2)
    time_since_last_transaction_minutes = np.round(rng.exponential(300, size=rows), 2)

    # MERCHANT RISK
    merchant_risk_score = np.round(rng.beta(2, 8, size=rows) * 100, 2)
    merchant_transaction_volume = rng.lognormal(np.log(5000), 1.0, size=rows).astype(
        int
    )

    # FRAUD SCORE (continuous)
    fraud_score = (
        -6.2
        + 0.0020 * transaction_amount
        + 0.70 * new_device_flag
        + 0.35 * failed_transactions_24h
        + 0.025 * distance_from_home_km
        + 0.15 * transactions_last_1h
        + 0.008 * merchant_risk_score
        + 0.0005 * customer_transaction_count_30d
        + np.where(hour < 5, 1.0, 0)
        + np.where(hour > 22, 0.4, 0)
    )
    fraud_score = np.clip(fraud_score, -20, 20)
    fraud_probability = 1 / (1 + np.exp(-fraud_score))

    # >>> Balanced binary labels via top-k threshold
    is_fraud = balance_labels(fraud_probability, TARGET_RATE)

    df = pd.DataFrame(
        {
            "transaction_id": transaction_ids,
            "customer_id": customer_ids,
            "transaction_timestamp": timestamp_str,
            "customer_age": age,
            "customer_gender": gender,
            "customer_income": customer_income,
            "account_age_days": account_age_days,
            "customer_segment": customer_segment,
            "home_country": country,
            "home_city": city,
            "transaction_amount": transaction_amount,
            "currency": "USD",
            "payment_method": payment_method,
            "merchant_id": merchant_ids,
            "merchant_category": merchant_category,
            "device_id": device_ids,
            "device_type": device_type,
            "browser": browser,
            "operating_system": operating_system,
            "transactions_last_1h": transactions_last_1h,
            "transactions_last_24h": transactions_last_24h,
            "avg_transaction_amount_30d": avg_transaction_amount_30d,
            "customer_transaction_count_30d": customer_transaction_count_30d,
            "failed_transactions_24h": failed_transactions_24h,
            "new_device_flag": new_device_flag,
            "new_merchant_flag": new_merchant_flag,
            "distance_from_home_km": distance_from_home_km,
            "time_since_last_transaction_minutes": time_since_last_transaction_minutes,
            "merchant_risk_score": merchant_risk_score,
            "merchant_transaction_volume": merchant_transaction_volume,
            "is_fraud": is_fraud,
        }
    )

    return df


# DATA CORRUPTION
def corrupt_data(df):
    print(">> INJECTING REALISTIC DATA-QUALITY PROBLEMS... <<")

    # Keep an un-corrupted copy of is_fraud to ensure labels stay intact
    fraud_labels = df["is_fraud"].copy()

    inject_missing_values(df, "customer_income", 0.04)
    inject_missing_values(df, "device_id", 0.03)
    inject_missing_values(df, "merchant_id", 0.02)
    inject_missing_values(df, "merchant_category", 0.02)

    inject_categorical_inconsistency(df, "payment_method", 0.02)
    inject_categorical_inconsistency(df, "home_country", 0.015)
    inject_categorical_inconsistency(df, "device_type", 0.015)

    inject_outliers(df, "transaction_amount", 0.01, multiplier=20)
    inject_outliers(df, "customer_income", 0.005, multiplier=8)

    # Invalid numerical values
    invalid_count = max(1, int(len(df) * 0.002))
    invalid_indices = rng.choice(len(df), size=invalid_count, replace=False)
    df.loc[invalid_indices, "transaction_amount"] = -1

    # Zero values
    zero_count = max(1, int(len(df) * 0.001))
    zero_indices = rng.choice(len(df), size=zero_count, replace=False)
    df.loc[zero_indices, "transaction_amount"] = 0

    # Duplicate rows (small %). Timestamp is regenerated on the dupes so they
    # aren't EXACT duplicates (more realistic for a data-quality scenario).
    duplicate_count = max(1, int(len(df) * 0.01))
    duplicate_indices = rng.choice(len(df), size=duplicate_count, replace=False)
    duplicate_rows = df.iloc[duplicate_indices].copy()

    duplicate_rows["transaction_timestamp"] = (
        pd.to_datetime(duplicate_rows["transaction_timestamp"])
        + pd.to_timedelta(rng.integers(0, 3600, size=len(duplicate_rows)), unit="s")
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    df = pd.concat([df, duplicate_rows], ignore_index=True)

    # Duplicate transaction IDs (data-entry style collisions)
    id_duplicate_count = max(1, int(len(df) * 0.005))
    duplicate_id_indices = rng.choice(len(df), size=id_duplicate_count, replace=False)
    source_indices = rng.choice(len(df), size=id_duplicate_count, replace=False)

    df.loc[duplicate_id_indices, "transaction_id"] = df.loc[
        source_indices, "transaction_id"
    ].values

    # Ensure is_fraud survived corruption (duplicate rows appended above
    # carry their own labels; but re-align to be safe).
    df["is_fraud"] = df["is_fraud"].astype(int)

    # Shuffle
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    return df


# GENERATE DATASET
def generate_dataset():
    print("\n" + "=" * 40)
    print(">> FRAUD DETECTION DATASET GENERATOR <<")
    print("=" * 40)
    print(f":> ROWS REQUESTED : {ROW_COUNT:,}")
    print(f":> CHUNK SIZE     : {CHUNK_SIZE:,}")
    print(f":> TARGET FRAUD % : {TARGET_RATE * 100:.2f}%")
    print(f":> OUTPUT FILE    : {OUTPUT_FILE}")
    print(f":> RANDOM SEED    : {RANDOM_SEED}")

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    total_written = 0
    first_chunk = True
    number_of_chunks = math.ceil(ROW_COUNT / CHUNK_SIZE)

    for chunk_number in range(number_of_chunks):
        remaining = ROW_COUNT - total_written
        rows = min(CHUNK_SIZE, remaining)

        print(f":> CHUNK {chunk_number + 1}/{number_of_chunks} ({rows:,} ROWS)...")

        df = generate_chunk(total_written, rows)
        df = corrupt_data(df)

        df.to_csv(
            OUTPUT_FILE,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )

        total_written += rows
        first_chunk = False
        print(f":> PROGRESS: {total_written:,}/{ROW_COUNT:,}")

    print("=" * 40)
    print(">> DATA GENERATION COMPLETED!")
    print("=" * 40)

    validate_dataset()


# VALIDATION
def validate_dataset():
    print(">> VALIDATING GENERATED CSV...")

    # Sample-based validation (memory-safe)
    df = pd.read_csv(OUTPUT_FILE, nrows=100_000)

    print("\n>> COLUMNS:")
    print(df.columns.tolist())

    print("\n>> DATA TYPES:")
    print(df.dtypes)

    print("\n>> MISSING VALUES (top 10):")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    print("\n>> DUPLICATE ROWS (sample):")
    print(df.duplicated().sum())

    print("\n>> DUPLICATE TRANSACTION IDs (sample):")
    print(df["transaction_id"].duplicated().sum())

    print("\n>> FRAUD DISTRIBUTION (sample):")
    print(df["is_fraud"].value_counts(normalize=True).sort_index())

    print("\n>> TRANSACTION AMOUNT STATISTICS (sample):")
    print(df["transaction_amount"].describe())

    # Full-file fraud distribution (streaming, so it stays balanced across chunks)
    fraud_total = 0
    row_total = 0

    for chunk in pd.read_csv(OUTPUT_FILE, usecols=["is_fraud"], chunksize=200_000):
        fraud_total += int(chunk["is_fraud"].sum())
        row_total += len(chunk)

    full_fraud_rate = fraud_total / row_total if row_total else 0.0

    # Write report
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(">> FRAUD DATASET QUALITY REPORT <<\n")
        f.write("=" * 50 + "\n")
        f.write(f":> ROWS REQUESTED      : {ROW_COUNT:,}\n")
        f.write(f":> VALIDATION SAMPLE   : {len(df):,}\n")
        f.write(f":> TARGET FRAUD RATE   : {TARGET_RATE:.4f}\n")
        f.write(f":> FULL-FILE FRAUD RATE: {full_fraud_rate:.4f}\n")
        f.write(f":> FULL ROWS WRITTEN   : {row_total:,}\n")
        f.write("=" * 50 + "\n")

        f.write("\n:> MISSING VALUES (sample):\n")
        f.write(str(df.isnull().sum()) + "\n")

        f.write(f"\n:> DUPLICATE ROWS (sample)        : {df.duplicated().sum()}\n")
        f.write(
            f":> DUPLICATE TXN IDs (sample)     : {df['transaction_id'].duplicated().sum()}\n"
        )

        f.write("\n:> FRAUD DISTRIBUTION (sample):\n")
        f.write(str(df["is_fraud"].value_counts(normalize=True)) + "\n")

        f.write("\n:> FRAUD DISTRIBUTION (full file):\n")
        f.write(f"  fraud = {fraud_total:,}\n")
        f.write(f"  legit = {row_total - fraud_total:,}\n")
        f.write(f"  rate  = {full_fraud_rate:.4f}\n")

    print(f"\n:> QUALITY REPORT SAVED TO: {REPORT_FILE}")
    print(f":> CSV SAVED TO          : {OUTPUT_FILE}")


# MAIN
if __name__ == "__main__":
    generate_dataset()
