
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cleaned_retail.csv"


def check_data_drift():

    # Load cleaned dataset
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Select numerical features for drift analysis
    features = [
        "Quantity",
        "UnitPrice",
        "TotalTransactionValue"
    ]

    # Compare the first and last portions of the dataset
    split_index = len(df) // 2

    reference_data = df.iloc[:split_index]
    recent_data = df.iloc[split_index:]

    print("SALES DATA DRIFT REVIEW")
    print("-----------------------")

    for feature in features:

        reference_mean = reference_data[feature].mean()
        recent_mean = recent_data[feature].mean()

        change = (
            abs(recent_mean - reference_mean)
            / (abs(reference_mean) + 1e-10)
        ) * 100

        print(f"\nFeature: {feature}")
        print("Reference mean:", round(reference_mean, 4))
        print("Recent mean:", round(recent_mean, 4))
        print("Percentage change:", round(change, 2), "%")

        if change > 20:
            print("Observation: Significant change detected.")
        else:
            print("Observation: No major change detected.")


if __name__ == "__main__":
    check_data_drift()
