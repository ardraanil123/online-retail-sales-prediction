
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/
COPY data/cleaned_retail.csv ./data/cleaned_retail.csv

RUN mkdir -p /app/storage

ENV DB_PATH=/app/storage/predictions.db

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]
