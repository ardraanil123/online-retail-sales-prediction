
import pandas as pd
from pathlib import Path


# Get the main project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset location
DATA_PATH = BASE_DIR / "data" / "online_retail.xlsx"


def load_data():
    """Load the original Excel dataset."""
    df = pd.read_excel(DATA_PATH)

    print("Dataset loaded successfully!")
    print("Original dataset shape:", df.shape)

    return df


def clean_data(df):
    """Clean the retail transaction dataset."""

    df = df.copy()

    # Remove extra spaces from column names
    df.columns = df.columns.str.strip()

    # Convert columns into appropriate data types
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )
    df["Quantity"] = pd.to_numeric(
        df["Quantity"], errors="coerce"
    )
    df["UnitPrice"] = pd.to_numeric(
        df["UnitPrice"], errors="coerce"
    )

    # Remove duplicate records
    df = df.drop_duplicates()

    # Remove cancelled transactions
    cancelled = (
        df["InvoiceNo"].astype(str).str.strip().str.upper().str.startswith("C")
    )
    df = df.loc[~cancelled].copy()

    # Handle missing product descriptions
    df["Description"] = df["Description"].fillna("Unknown Product")

    # Remove rows with missing essential values
    essential_columns = [
        "InvoiceNo",
        "StockCode",
        "Quantity",
        "UnitPrice",
        "InvoiceDate",
        "Country"
    ]

    df = df.dropna(subset=essential_columns)

    # Remove invalid quantity and price values
    df = df[
        (df["Quantity"] > 0) &
        (df["UnitPrice"] > 0)
    ]

    df = df.reset_index(drop=True)

    print("Data cleaning completed!")
    print("Cleaned dataset shape:", df.shape)

    return df


def add_features(df):
    """Create additional features for analysis."""

    df = df.copy()

    # Calculate total transaction value
    df["TotalTransactionValue"] = (
        df["Quantity"] * df["UnitPrice"]
    )

    # Extract date-related features
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["Day"] = df["InvoiceDate"].dt.day
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_of_week

    # Create year-month feature for sales trends
    df["YearMonth"] = (
        df["InvoiceDate"].dt.to_period("M").astype(str)
    )

    print("Feature engineering completed!")

    return df


def prepare_data():
    """Complete preprocessing pipeline."""

    df = load_data()
    df = clean_data(df)
    df = add_features(df)

    # Save the processed dataset
    output_path = BASE_DIR / "data" / "cleaned_retail.csv"
    df.to_csv(output_path, index=False)

    print("Processed dataset saved successfully!")
    print("Final dataset shape:", df.shape)

    return df


if __name__ == "__main__":
    prepare_data()
