

from pathlib import Path
import logging
import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "retail_model_v1.joblib"

# Create a folder to store application logs
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=LOG_DIR / "project.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def load_model():
    """Load the trained machine learning pipeline."""
    logger.info("Loading trained model")
    model = joblib.load(MODEL_PATH)
    return model


def predict_sales(quantity, unit_price, month, day, day_of_week, country):
    """Predict total transaction value."""

    try:
        model = load_model()

        input_data = pd.DataFrame([{
            "Quantity": quantity,
            "UnitPrice": unit_price,
            "Month": month,
            "Day": day,
            "DayOfWeek": day_of_week,
            "Country": country
        }])

        prediction = model.predict(input_data)
        result = float(prediction[0])

        logger.info("Prediction completed successfully: %.2f", result)

        return result

    except Exception:
        logger.exception("Error occurred during sales prediction")
        raise


if __name__ == "__main__":

    result = predict_sales(
        quantity=5,
        unit_price=10,
        month=11,
        day=26,
        day_of_week=3,
        country="United Kingdom"
    )

    print("Predicted transaction value:", round(result, 2))

