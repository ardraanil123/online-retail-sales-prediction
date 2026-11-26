# Online Retail Sales Prediction and Product Analysis

## 1. Project Overview

This project focuses on predicting the expected transaction value of online retail transactions using Machine Learning. It uses the UCI Online Retail dataset and implements an Extra Trees Regression model.

The project also provides an interactive Streamlit application for sales prediction, sales analysis, transaction filtering, and prediction history management.

## 2. Objectives

- Develop a machine learning model to estimate transaction values.
- Perform data cleaning and feature engineering.
- Analyse sales trends, products, and countries.
- Build an interactive Streamlit application.
- Store prediction records using an SQLite database.
- Track machine learning experiments using MLflow.
- Containerize the application using Docker.
- Perform automated testing and basic data-drift analysis.

## 3. Dataset

**Dataset:** Individual Online Retail Dataset  
**Source:** UCI Machine Learning Repository  
**Dataset ID:** 352

The dataset contains online retail transaction information, including invoice numbers, product descriptions, quantities, unit prices, invoice dates, and countries.

Dataset source: https://archive.ics.uci.edu/dataset/352/online+retail

## 4. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Extra Trees Regressor
- Streamlit
- SQLite
- Joblib
- MLflow
- Pytest
- Docker and Docker Compose
- Matplotlib

## 5. Machine Learning Methodology

The project uses an Extra Trees Regressor for transaction value estimation.

### Input Features
- Quantity
- Unit Price
- Month
- Day
- Day of Week
- Country

### Target Variable

TotalTransactionValue = Quantity × UnitPrice

Country is encoded using One-Hot Encoding.

The dataset is divided into 80% training data and 20% testing data. A training sample of 150,000 records is used to reduce training time.

### Model Performance

| Evaluation Metric | Result |
|---|---:|
| MAE | 0.6781 |
| RMSE | 32.6191 |
| R² Score | 0.8417 |

The complete preprocessing and prediction pipeline is saved using Joblib.

## 6. Application Features

The Streamlit application provides:

1. **Sales Prediction:** Enter transaction details and obtain an estimated transaction value.
2. **CSV Upload:** Upload transaction data and generate multiple predictions.
3. **Sales Analysis:** View monthly sales trends, top products, and country-wise sales.
4. **Transaction Filtering:** Search and filter transaction records.
5. **Prediction History:** Store and view previous predictions using SQLite.

## 7. MLOps Implementation

The project incorporates the following MLOps practices:

- **Experiment Tracking:** MLflow records model parameters and evaluation metrics.
- **Model Saving:** Joblib stores the complete trained pipeline.
- **Model Versioning:** The application records the model version associated with predictions.
- **Automated Testing:** Pytest is used to validate data preprocessing and feature engineering.
- **Containerization:** Docker and Docker Compose package and run the application.
- **Operational Logging:** Prediction activities and errors are recorded.
- **Data Drift Review:** A basic comparison of numerical feature averages is performed to identify possible changes in sales data.

## 8. Project Structure

```text
Online_retail_project/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── online_retail.xlsx
│   ├── cleaned_retail.csv
│   └── predictions.db
│
├── models/
│   ├── retail_model_v1.joblib
│   └── model_comparison.csv
│
├── src/
│   ├── database.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── compare_models.py
│   └── drift_check.py
│
├── tests/
│   └── test_project.py
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── README.md
```

## 9. Running the Application

### Using Docker Compose

1. Install Docker Desktop.
2. Open the project directory in the terminal.
3. Build and start the application:

```bash
docker compose up --build -d
```

4. Open the application in a browser:

```text
http://localhost:8501
```

5. To stop the application:

```bash
docker compose down
```

### Running Automated Tests

Execute:

```bash
pytest -v
```

### Running Data Drift Analysis

Execute:

```bash
python src/drift_check.py
```

### Running Model Training

Execute:

```bash
python src/train.py
```

## 10. Limitations

- The model is trained on historical transaction data.
- Prediction performance depends on the quality of the input data.
- The drift analysis uses a basic comparison of feature averages and is not a comprehensive statistical drift test.
- The model estimates transaction value rather than forecasting future customer demand.

## 11. Future Scope

- Implement automated CI workflows using GitHub Actions.
- Improve data-drift detection and monitoring.
- Explore additional machine learning algorithms.
- Extend the application with more advanced sales analytics.
- Deploy the application to a cloud platform.

## 12. Conclusion

The project demonstrates the application of machine learning in online retail transaction analysis. It integrates an Extra Trees Regression model with an interactive Streamlit application, SQLite database storage, MLflow experiment tracking, automated testing, Docker containerization, and basic data-drift analysis.