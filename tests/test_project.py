
import sys
from pathlib import Path

import pandas as pd

# Add project root to Python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.preprocessing import clean_data, add_features


def sample_data():
    """Create sample retail transactions for testing."""
    return pd.DataFrame({
        "InvoiceNo": ["1001", "C1002", "1003", "1004"],
        "StockCode": ["P01", "P02", "P03", "P04"],
        "Description": ["Product A", "Product B", None, "Product D"],
        "Quantity": [5, 2, 3, 4],
        "InvoiceDate": [
            "2011-11-26",
            "2011-11-26",
            "2011-11-27",
            "2011-11-28"
        ],
        "UnitPrice": [10, 5, 8, 6],
        "Country": [
            "United Kingdom",
            "France",
            "Germany",
            "Spain"
        ]
    })


# Existing project tests

def test_cleaned_data_file_exists():
    assert (BASE_DIR / "data" / "cleaned_retail.csv").exists()


def test_cleaned_data_not_empty():
    df = pd.read_csv(
        BASE_DIR / "data" / "cleaned_retail.csv",
        low_memory=False
    )
    assert not df.empty


def test_cleaned_data_has_required_columns():
    df = pd.read_csv(
        BASE_DIR / "data" / "cleaned_retail.csv",
        low_memory=False
    )

    required_columns = [
        "Quantity",
        "UnitPrice",
        "TotalTransactionValue"
    ]

    for column in required_columns:
        assert column in df.columns


def test_transaction_value_calculation():
    quantity = 5
    unit_price = 10
    assert quantity * unit_price == 50


# Required preprocessing tests

def test_cancelled_transactions_are_removed():
    df = clean_data(sample_data())

    assert not df["InvoiceNo"].str.startswith("C").any()


def test_duplicate_transactions_are_removed():
    df = sample_data()

    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    cleaned = clean_data(df)

    assert not cleaned.duplicated().any()


def test_missing_product_description_is_handled():
    df = clean_data(sample_data())

    assert "Unknown Product" in df["Description"].values


def test_date_derived_features():
    df = clean_data(sample_data())
    featured = add_features(df)

    first_row = featured.iloc[0]

    assert first_row["Year"] == 2011
    assert first_row["Month"] == 11
    assert first_row["Day"] == 26
    assert first_row["DayOfWeek"] == 5
    assert first_row["YearMonth"] == "2011-11"
