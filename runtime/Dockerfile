FROM python:3.11-slim
WORKDIR /app
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pipeline ./pipeline
COPY fraud_detector ./fraud_detector
COPY data_loader ./data_loader
COPY scoring_writer ./scoring_writer
COPY transaction_simulator ./transaction_simulator
COPY scripts ./scripts
COPY tests ./tests
COPY pytest.ini .
