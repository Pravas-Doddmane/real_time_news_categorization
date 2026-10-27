from __future__ import annotations

import argparse
import json
import sys
import time

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

from common import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC, make_article

SAMPLES = [
    {"title": "City club wins national football final", "text": "The team lifted the trophy after scoring two goals in the final match. The coach praised the players and supporters after a strong season."},
    {"title": "New chip improves laptop battery performance", "text": "Technology researchers announced a processor design that improves computing performance and reduces power consumption in laptops and mobile devices."},
    {"title": "Parliament debates new public policy", "text": "Lawmakers debated the proposed bill in parliament today. Opposition members questioned the policy while government ministers defended the legislation."},
    {"title": "Markets rise as company earnings improve", "text": "Shares rose after several companies reported stronger quarterly profits. Investors watched market prices, revenue growth and the central bank outlook."},
    {"title": "Film festival announces its award winners", "text": "The annual film festival announced its award winners, with actors and directors celebrating the films selected by the judging panel."},
]


def create_producer():
    last_error = None
    for attempt in range(10):
        try:
            return KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
                value_serializer=lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8"),
                key_serializer=lambda key: key.encode("utf-8") if key else None,
                retries=5, request_timeout_ms=20000, api_version_auto_timeout_ms=10000
            )
        except NoBrokersAvailable as exc:
            last_error = exc
            print(f"Kafka not ready (attempt {attempt + 1}/10); retrying in 3 seconds...")
            time.sleep(3)
    raise RuntimeError(f"Could not connect to Kafka at {KAFKA_BOOTSTRAP_SERVERS}: {last_error}")


def load_dataset_articles(limit: int = 10, category: str | None = None) -> list[dict]:
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "dataset" / "BBC",
        root / "dataset" / "BCC",
        root / "dataset",
        root / "data" / "BBC",
    ]
    raw_dir = None
    for c in candidates:
        if c.is_dir() and any((c / cat).is_dir() for cat in ["business", "tech", "sport", "politics", "entertainment"]):
            raw_dir = c
            break

    articles = []
    if raw_dir:
        cat_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
        if category:
            cat_dirs = [d for d in cat_dirs if d.name.lower() == category.lower()]
        cat_files = {d.name.lower(): sorted(d.glob("*.txt")) for d in cat_dirs}
        idx = 0
        while True:
            added_any = False
            for cat_name, files in cat_files.items():
                if idx < len(files):
                    fpath = files[idx]
                    try:
                        content = fpath.read_text(encoding="utf-8", errors="replace").strip()
                    except Exception:
                        content = fpath.read_text(encoding="latin-1", errors="replace").strip()
                    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
                    title = lines[0] if lines else f"{cat_name.title()} article {fpath.stem}"
                    body = " ".join(lines[1:]) if len(lines) > 1 else title
                    articles.append(make_article(title, body, f"dataset-{cat_name}"))
                    added_any = True
                    if limit > 0 and len(articles) >= limit:
                        return articles
            if not added_any:
                break
            idx += 1
        return articles

    csv_path = root / "data" / "bbc-text.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        if category:
            df = df[df["category"].str.lower() == category.lower()]
        for _, row in df.head(limit if limit > 0 else 10).iterrows():
            text = str(row["text"])
            title = text[:80] + ("..." if len(text) > 80 else "")
            articles.append(make_article(title, text, f"dataset-{row.get('category', 'unknown')}"))
    return articles


def main():
    parser = argparse.ArgumentParser(description="Publish news articles to Kafka.")
    parser.add_argument("--sample", action="store_true", help="Send five sample articles")
    parser.add_argument("--dataset", action="store_true", help="Stream articles from the BBC dataset folder")
    parser.add_argument("--limit", type=int, default=10, help="Number of articles to send from dataset (default: 10)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay in seconds between streaming articles (default: 1.0)")
    parser.add_argument("--category", help="Optional category filter when using --dataset (business, entertainment, politics, sport, tech)")
    parser.add_argument("--title", help="Article title")
    parser.add_argument("--text", help="Article body text")
    parser.add_argument("--source", default="manual", help="Article source label")
    args = parser.parse_args()

    delay = 0.0
    if args.sample:
        articles = [make_article(item["title"], item["text"], "sample") for item in SAMPLES]
    elif args.dataset:
        articles = load_dataset_articles(limit=args.limit, category=args.category)
        if not articles:
            parser.error("No articles found in dataset/BBC or data/bbc-text.csv.")
        delay = max(args.delay, 0.0)
    elif args.title and args.text:
        articles = [make_article(args.title, args.text, args.source)]
    else:
        parser.error("Use --sample, --dataset, or provide both --title and --text.")

    producer = create_producer()
    try:
        total = len(articles)
        for idx, article in enumerate(articles, start=1):
            metadata = producer.send(KAFKA_TOPIC, key=article["article_id"], value=article).get(timeout=30)
            print(f"[{idx}/{total}] Published '{article['title'][:60]}' (id={article['article_id']}) to {metadata.topic}, partition={metadata.partition}, offset={metadata.offset}")
            if delay > 0 and idx < total:
                time.sleep(delay)
    finally:
        producer.flush(timeout=20)
        producer.close()



if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
