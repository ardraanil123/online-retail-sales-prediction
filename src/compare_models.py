
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cleaned_retail.csv"


# Load dataset
df = pd.read_csv(DATA_PATH, low_memory=False)

print("Dataset loaded:", df.shape)


# Select features and target
features = [
    "Quantity",
    "UnitPrice",
    "Month",
    "Day",
    "DayOfWeek",
    "Country"
]

X = df[features]
y = df["TotalTransactionValue"]


# Use the same train-test split for every model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Use the same training sample for fair comparison
if len(X_train) > 150000:
    X_train = X_train.sample(n=150000, random_state=42)
    y_train = y_train.loc[X_train.index]


# Preprocessing
def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "country_encoder",
                OneHotEncoder(handle_unknown="ignore"),
                ["Country"]
            )
        ],
        remainder="passthrough"
    )


# Models to compare
models = {
    "Random Forest - Baseline": RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ),

    "Random Forest - Tuned 1": RandomForestRegressor(
        n_estimators=150,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Random Forest - Tuned 2": RandomForestRegressor(
        n_estimators=150,
        max_depth=30,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=150,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}


results = []

print("\nStarting model comparison...\n")

for name, regressor in models.items():

    print("Training:", name)

    pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("regressor", regressor)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))
    print("-" * 40)


# Display comparison
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="R2 Score", ascending=False
)

print("\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))

# Save comparison results
output_path = BASE_DIR / "models" / "model_comparison.csv"
results_df.to_csv(output_path, index=False)

print("\nComparison results saved to:", output_path)
