"""Kafka transactions → CatBoost → Kafka scoring."""

import json
import logging
import os
import pandas as pd
from confluent_kafka import Consumer, Producer
from fraud_detector.src.preprocessing import run_preproc
from fraud_detector.src.scorer import ensure_bootstrap, make_pred

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingService:
    def __init__(self):
        brokers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
        self.consumer = Consumer({"bootstrap.servers": brokers, "group.id": "ml-scorer",
                                  "auto.offset.reset": "earliest", "enable.auto.commit": False})
        self.consumer.subscribe([os.getenv("KAFKA_TRANSACTIONS_TOPIC", "transactions")])
        self.producer = Producer({"bootstrap.servers": brokers, "enable.idempotence": True})

    def process_message(self, msg):
        data = json.loads(msg.value())
        processed = run_preproc(pd.DataFrame([data["data"]]))
        submission, _, version = make_pred(processed, "kafka_stream")
        submission["transaction_id"] = data["transaction_id"]
        submission["model_version"] = version
        errors = []
        self.producer.produce(
            os.getenv("KAFKA_SCORING_TOPIC", "scoring"), key=data["transaction_id"],
            value=submission.to_json(orient="records"),
            on_delivery=lambda error, message: errors.append(error) if error else None,
        )
        remaining = self.producer.flush(30)
        if remaining or errors:
            raise RuntimeError("Не удалось отправить результат в Kafka")
        self.consumer.commit(message=msg, asynchronous=False)

    def process_messages(self):
        try:
            while True:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    raise RuntimeError(str(msg.error()))
                self.process_message(msg)
        finally:
            self.consumer.close()


if __name__ == "__main__":
    ensure_bootstrap()
    ProcessingService().process_messages()
