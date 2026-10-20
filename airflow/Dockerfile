FROM apache/airflow:2.10.5-python3.11
USER airflow
RUN pip install --no-cache-dir "apache-airflow==2.10.5" \
    "numpy==1.26.4" "pandas==2.1.4" "scipy==1.14.1" \
    "pyarrow==16.1.0" "scikit-learn==1.5.2" "catboost==1.2.8" \
    "psycopg2-binary==2.9.10" "cloudpickle==2.2.1"
USER root
COPY entrypoint.sh /custom-entrypoint.sh
RUN chmod +x /custom-entrypoint.sh
ENTRYPOINT ["/custom-entrypoint.sh"]
