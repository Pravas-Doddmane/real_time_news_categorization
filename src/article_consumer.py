from __future__ import annotations

import json
import sys
import time

import joblib
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

from common import (
    HDFS_WEBHDFS_URL, KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC, MODEL_PATH,
    hdfs_append_jsonl, hdfs_mkdir, utc_now
)


def create_consumer():
    last_error = None
    for attempt in range(10):
        try:
            return KafkaConsumer(
                KAFKA_TOPIC, bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
                group_id="news-categorization-consumer", auto_offset_reset="earliest",
                enable_auto_commit=True,
                value_deserializer=lambda value: json.loads(value.decode("utf-8")),
                consumer_timeout_ms=1000
            )
        except NoBrokersAvailable as exc:
            last_error = exc
            print(f"Kafka not ready (attempt {attempt + 1}/10); retrying in 3 seconds...")
            time.sleep(3)
    raise RuntimeError(f"Could not connect to Kafka: {last_error}")


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run `python src/train_model.py` first.")
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model: {MODEL_PATH}", flush=True)
    print(f"Kafka: {KAFKA_BOOTSTRAP_SERVERS}; topic: {KAFKA_TOPIC}", flush=True)
    print(f"HDFS WebHDFS: {HDFS_WEBHDFS_URL}", flush=True)
    hdfs_mkdir("/news/raw")
    hdfs_mkdir("/news/classified")
    consumer = create_consumer()
    print("Consumer running. Press Ctrl+C to stop.", flush=True)
    try:
        while True:
            received = False
            for message in consumer:
                received = True
                article = message.value
                title = str(article.get("title", ""))
                body = str(article.get("text", ""))
                if not (title + body).strip():
                    print("Skipping article with no title or text.")
                    continue
                raw_record = {
                    **article, "kafka_topic": message.topic,
                    "kafka_partition": message.partition, "kafka_offset": message.offset,
                    "received_at": utc_now()
                }
                category = str(model.predict([(title + " " + body).strip()])[0])
                classified_record = {
                    **raw_record, "predicted_category": category,
                    "model_name": "TF-IDF + LinearSVC", "classified_at": utc_now()
                }
                article_id = str(article.get("article_id", f"{message.partition}-{message.offset}"))
                safe_id = "".join(ch for ch in article_id if ch.isalnum() or ch in "-_")[:100]
                try:
                    hdfs_append_jsonl(f"/news/raw/{safe_id}.jsonl", raw_record)
                    hdfs_append_jsonl(f"/news/classified/{safe_id}.jsonl", classified_record)
                    print(f"\nARTICLE: {title}\nPREDICTED CATEGORY: {category}\nSAVED: /news/classified/{safe_id}.jsonl", flush=True)
                except Exception as exc:
                    print(f"HDFS write failed for {article_id}: {exc}", file=sys.stderr, flush=True)
                    print(f"Prediction still produced: {category}", flush=True)
            if not received:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping consumer...")
    finally:
        consumer.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
