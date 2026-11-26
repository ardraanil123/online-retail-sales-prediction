
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cleaned_retail.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "retail_model_v1.joblib"

MODEL_DIR.mkdir(exist_ok=True)


def train_model():

    # Load cleaned dataset
    df = pd.read_csv(
    DATA_PATH,
    low_memory=False,
    dtype={"InvoiceNo": "string"}
)

    print("Cleaned dataset loaded:", df.shape)

    # Select input features and target
    features = [
        "Quantity",
        "UnitPrice",
        "Month",
        "Day",
        "DayOfWeek",
        "Country"
    ]

    target = "TotalTransactionValue"

    X = df[features]
    y = df[target]

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Use a sample of training data to reduce training time
    if len(X_train) > 150000:
        X_train = X_train.sample(
            n=150000,
            random_state=42
        )
        y_train = y_train.loc[X_train.index]

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Encode categorical feature
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "country_encoder",
                OneHotEncoder(handle_unknown="ignore"),
                ["Country"]
            )
        ],
        remainder="passthrough"
    )

    # Create complete machine learning pipeline
    model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        ExtraTreesRegressor(
            n_estimators=50,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    )
    ])

    # MLflow experiment tracking
    mlflow.set_experiment("Online_Retail_Sales_Prediction")

    with mlflow.start_run():

        # Train model
        print("\nTraining Extra tree model...")
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Evaluate model
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        print("\nModel Evaluation Results")
        print("------------------------")
        print("MAE:", round(mae, 4))
        print("RMSE:", round(rmse, 4))
        print("R2 Score:", round(r2, 4))

        # Save complete pipeline
        joblib.dump(model, MODEL_PATH)

        print("\nModel saved successfully!")
        print("Model location:", MODEL_PATH)

        # Log parameters and metrics
        mlflow.log_param("model_type", "ExtraTreesRegressor")
        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("max_depth", "None")
        mlflow.log_param("min_samples_leaf", 2)
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_param("testing_samples", len(X_test))
        mlflow.log_param("dataset_version", "UCI_Online_Retail_352")

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2_Score", r2)

        mlflow.sklearn.log_model(
    model,
    name="retail_sales_pipeline",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

        print("MLflow experiment logged successfully!")

    return model


if __name__ == "__main__":
    train_model()
