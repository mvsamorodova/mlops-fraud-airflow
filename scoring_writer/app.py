"""Сохранение результатов Kafka в PostgreSQL."""

import json
import logging
import os
from confluent_kafka import Consumer
from pipeline.db import get_conn

logging.basicConfig(level=logging.INFO)


def create_table(conn):
    with conn, conn.cursor() as cur:
        cur.execute("""CREATE TABLE IF NOT EXISTS scores (
            id SERIAL PRIMARY KEY, transaction_id TEXT UNIQUE NOT NULL,
            score FLOAT NOT NULL, fraud_flag INT NOT NULL,
            model_version TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")


def insert_score(conn, data):
    with conn, conn.cursor() as cur:
        cur.execute("""INSERT INTO scores(transaction_id, score, fraud_flag, model_version)
            VALUES (%s,%s,%s,%s) ON CONFLICT(transaction_id) DO NOTHING""",
            (data["transaction_id"], data["score"], data["fraud_flag"], data["model_version"]))


def run_consumer():
    consumer = Consumer({
        "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"),
        "group.id": "scoring-writer", "auto.offset.reset": "earliest", "enable.auto.commit": False,
    })
    consumer.subscribe([os.getenv("KAFKA_SCORING_TOPIC", "scoring")])
    conn = get_conn()
    create_table(conn)
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                raise RuntimeError(str(msg.error()))
            for record in json.loads(msg.value()):
                insert_score(conn, record)
            consumer.commit(message=msg, asynchronous=False)
    finally:
        consumer.close()
        conn.close()


if __name__ == "__main__":
    run_consumer()
