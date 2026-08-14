import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "orders_dataset.csv"

df = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("DATASET VERIFICATION")
print("=" * 60)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\nColumns:")
for column in df.columns:
    print(f"  - {column}")

print("\nOverall return rate:")
print(f"{df['returned'].mean():.4%}")

print("\nMissing rating_given:")
missing_pct = df["rating_given"].isna().mean()
print(f"{missing_pct:.4%}")

print("\nReturn rate by product category:")
category_rates = (
    df.groupby("product_category")["returned"]
    .mean()
    .sort_values(ascending=False)
)

print(category_rates.to_string())

print("\nReturn rate by payment method:")
payment_rates = (
    df.groupby("payment_method")["returned"]
    .mean()
    .sort_values(ascending=False)
)

print(payment_rates.to_string())

print("\nMissing rating_given by payment method:")

missing_by_payment = (
    df.groupby("payment_method")["rating_given"]
    .apply(lambda x: x.isna().mean())
)

print(
    missing_by_payment
    .sort_values(ascending=False)
    .to_string()
)

cod_missing = missing_by_payment["COD"]

non_cod_missing = (
    df.loc[df["payment_method"] != "COD", "rating_given"]
    .isna()
    .mean()
)

print("\nMAR analysis:")
print(f"COD missing rate: {cod_missing:.4%}")
print(f"Non-COD missing rate: {non_cod_missing:.4%}")
print(
    f"Gap: {(cod_missing - non_cod_missing):.4%}"
)

print("\nExpected missingness classification: MAR")
print(
    "Reason: rating_given missingness depends on the observed "
    "payment_method column. It is therefore not MCAR, and it "
    "does not depend on the unobserved rating value itself, "
    "so it is not MNAR."
)