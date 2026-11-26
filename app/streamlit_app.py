
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Add project directory to Python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.predict import predict_sales
from src.database import (
    create_table,
    save_prediction,
    get_prediction_history
)


# Page configuration
st.set_page_config(
    page_title="Online Retail Sales Prediction",
    page_icon="🛍️",
    layout="wide"
)

# Initialize database
create_table()

# Application title
st.title("🛍️ Online Retail Sales Prediction")
st.write(
    "Predict transaction values and analyse online retail sales."
)


# Sidebar navigation
menu = st.sidebar.selectbox(
    "Navigation",
    [
        "Sales Prediction",
        "Sales Analysis",
        "Prediction History"
    ]
)


# Load cleaned dataset
DATA_PATH = BASE_DIR / "data" / "cleaned_retail.csv"


if menu == "Sales Prediction":

    st.header("Transaction Value Prediction")

    col1, col2 = st.columns(2)

    with col1:
        quantity = st.number_input(
            "Quantity",
            min_value=1,
            value=5
        )

        unit_price = st.number_input(
            "Unit Price (£)",
            min_value=0.01,
            value=10.0
        )

        month = st.selectbox(
            "Month",
            list(range(1, 13)),
            index=10
        )

    with col2:
        day = st.selectbox(
            "Day",
            list(range(1, 32)),
            index=25
        )

        day_of_week = st.selectbox(
            "Day of Week",
            list(range(7))
        )

        country = st.selectbox(
            "Country",
            [
                "United Kingdom",
                "France",
                "Germany",
                "EIRE",
                "Spain",
                "Netherlands",
                "Belgium",
                "Other"
            ]
        )

    if st.button("Predict Transaction Value"):

        try:
            result = predict_sales(
                quantity,
                unit_price,
                month,
                day,
                day_of_week,
                country
            )

            st.success(
                f"Predicted Transaction Value: £{result:.2f}"
            )

            save_prediction(
                quantity,
                unit_price,
                country,
                result,
                "ExtraTrees_v1"
            )

            st.info("Prediction saved successfully!")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.divider()
    st.subheader("Upload Transaction CSV")

    st.write(
        "Upload a CSV file containing transaction details "
        "to generate predictions in bulk."
    )

    uploaded_file = st.file_uploader(
        "Choose transaction CSV file",
        type=["csv"],
        key="transaction_upload"
    )

    if uploaded_file is not None:

        if st.button("Predict Uploaded Transactions"):

            try:
                upload_df = pd.read_csv(uploaded_file)

                required_columns = [
                    "Quantity",
                    "UnitPrice",
                    "Country"
                ]

                missing_columns = [
                    col for col in required_columns
                    if col not in upload_df.columns
                ]

                if missing_columns:
                    st.error(
                        f"Missing required columns: {missing_columns}"
                    )

                elif not (
                    all(col in upload_df.columns for col in
                        ["Month", "Day", "DayOfWeek"])
                    or "InvoiceDate" in upload_df.columns
                    or "Date" in upload_df.columns
                ):
                    st.error(
                        "Provide Month, Day and DayOfWeek, "
                        "or an InvoiceDate/Date column."
                    )

                elif upload_df.empty:
                    st.warning("The uploaded CSV file is empty.")

                elif len(upload_df) > 1000:
                    st.warning(
                        "Please upload a file containing at most "
                        "1000 transactions at a time."
                    )

                else:
                    if "Month" not in upload_df.columns:
                        date_column = (
                            "InvoiceDate"
                            if "InvoiceDate" in upload_df.columns
                            else "Date"
                        )

                        dates = pd.to_datetime(
                            upload_df[date_column],
                            errors="coerce"
                        )

                        upload_df["Month"] = dates.dt.month
                        upload_df["Day"] = dates.dt.day
                        upload_df["DayOfWeek"] = dates.dt.dayofweek

                    required_values = [
                        "Quantity",
                        "UnitPrice",
                        "Country",
                        "Month",
                        "Day",
                        "DayOfWeek"
                    ]

                    if upload_df[required_values].isnull().any().any():
                        st.error(
                            "Some transactions have missing or invalid "
                            "required values. Please correct the CSV."
                        )

                    else:
                        results = []

                        for _, row in upload_df.iterrows():

                            prediction = predict_sales(
                                int(row["Quantity"]),
                                float(row["UnitPrice"]),
                                int(row["Month"]),
                                int(row["Day"]),
                                int(row["DayOfWeek"]),
                                str(row["Country"])
                            )

                            save_prediction(
                                int(row["Quantity"]),
                                float(row["UnitPrice"]),
                                str(row["Country"]),
                                prediction,
                                "ExtraTrees_v1"
                            )

                            results.append(prediction)

                        upload_df["PredictedTransactionValue"] = results

                        st.success(
                            f"Successfully predicted "
                            f"{len(results)} transactions!"
                        )

                        st.dataframe(
                            upload_df,
                            use_container_width=True
                        )

                        st.info(
                            "All uploaded transaction predictions "
                            "have been saved to the database."
                        )

            except Exception as e:
                st.error(f"Upload prediction failed: {e}")


elif menu == "Sales Analysis":

    st.header("Sales Analysis Dashboard")

    if DATA_PATH.exists():

        df = pd.read_csv(DATA_PATH, low_memory=False)

        # Monthly sales trend
        monthly_sales = (
            df.groupby("YearMonth")["TotalTransactionValue"]
            .sum()
            .reset_index()
        )

        st.subheader("Monthly Sales Trend")

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(
            monthly_sales["YearMonth"],
            monthly_sales["TotalTransactionValue"],
            marker="o"
        )

        ax.set_xlabel("Month")
        ax.set_ylabel("Total Sales (£)")
        ax.tick_params(axis="x", rotation=45)

        st.pyplot(fig)
        plt.close(fig)

        # Top products
        st.subheader("Top 10 Products")

        top_products = (
            df.groupby("Description")["TotalTransactionValue"]
            .sum()
            .nlargest(10)
            .sort_values()
        )

        st.bar_chart(top_products)

        # Country-wise sales
        st.subheader("Country-wise Sales")

        country_sales = (
            df.groupby("Country")["TotalTransactionValue"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        st.bar_chart(country_sales)

        # Transaction filters
        st.subheader("Search and Filter Transactions")

        selected_country = st.selectbox(
            "Filter by Country",
            ["All"] + sorted(df["Country"].dropna().unique().tolist())
        )

        if selected_country != "All":
            df = df[df["Country"] == selected_country]

        search_product = st.text_input("Search Product")

        if search_product:
            df = df[
                df["Description"].str.contains(
                    search_product,
                    case=False,
                    na=False
                )
            ]

        st.dataframe(df.head(500), use_container_width=True)

    else:
        st.warning("Cleaned dataset not found.")


elif menu == "Prediction History":

    st.header("Prediction History")

    history = get_prediction_history()

    if history:

        columns = [
            "ID",
            "Timestamp",
            "Quantity",
            "Unit Price",
            "Country",
            "Predicted Value",
            "Model Version"
        ]

        history_df = pd.DataFrame(
            history,
            columns=columns
        )

        st.dataframe(
            history_df,
            use_container_width=True
        )

    else:
        st.info("No prediction history available yet.")


st.sidebar.markdown("---")
st.sidebar.caption("Online Retail ML Project | Extra Trees Regressor")
