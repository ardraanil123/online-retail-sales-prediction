
import sqlite3
import os
from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = os.getenv(
    "DB_PATH",
    str(BASE_DIR / "data" / "predictions.db")
)


def get_connection():
    """Create database connection."""
    return sqlite3.connect(DB_PATH)


def create_table():
    """Create predictions table."""

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            quantity REAL,
            unit_price REAL,
            country TEXT,
            predicted_value REAL,
            model_version TEXT
        )
    """)

    connection.commit()
    connection.close()


def save_prediction(
    quantity,
    unit_price,
    country,
    predicted_value,
    model_version="ExtraTrees_v1"
):
    """Save prediction details."""

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            timestamp,
            quantity,
            unit_price,
            country,
            predicted_value,
            model_version
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        quantity,
        unit_price,
        country,
        predicted_value,
        model_version
    ))

    connection.commit()
    connection.close()


def get_prediction_history():
    """Retrieve saved prediction history."""

    connection = get_connection()

    history = connection.execute("""
        SELECT * FROM predictions
        ORDER BY id DESC
    """).fetchall()

    connection.close()

    return history


if __name__ == "__main__":
    create_table()
    print("Database and predictions table created successfully!")
