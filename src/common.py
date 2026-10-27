from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import requests

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ROOT / "models" / "news_classifier.joblib")))
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "news-articles")
HDFS_WEBHDFS_URL = os.getenv("HDFS_WEBHDFS_URL", "http://localhost:9870/webhdfs/v1").rstrip("/")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_article(title: str, text: str, source: str = "demo") -> dict:
    return {
        "article_id": str(uuid.uuid4()),
        "title": title.strip(),
        "text": text.strip(),
        "source": source,
        "published_at": utc_now(),
    }


def hdfs_mkdir(path: str, retries: int = 10, delay: float = 3.0) -> None:
    url = f"{HDFS_WEBHDFS_URL}{path}"
    last_error = None
    for _ in range(retries):
        try:
            response = requests.put(url, params={"op": "MKDIRS"}, timeout=10, allow_redirects=True)
            if response.status_code in (200, 201):
                return
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(delay)
    raise RuntimeError(f"Could not create HDFS directory {path}. Last error: {last_error}")


def hdfs_append_jsonl(path: str, record: dict, retries: int = 10, delay: float = 2.0) -> None:
    # Store one JSON record per file to avoid append races.
    payload = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    url = f"{HDFS_WEBHDFS_URL}{path}"
    webhdfs_host = urlparse(HDFS_WEBHDFS_URL).hostname or "localhost"
    last_error = None
    for attempt in range(retries):
        try:
            # Step 1: Request upload location from NameNode without sending data payload
            init_resp = requests.put(
                url,
                params={"op": "CREATE", "overwrite": "false"},
                timeout=15,
                allow_redirects=False,
            )
            if init_resp.status_code == 307:
                loc = init_resp.headers.get("Location")
                if not loc:
                    raise RuntimeError("WebHDFS 307 redirect missing Location header")
                parsed_loc = urlparse(loc)
                # Map internal docker container hostname to WebHDFS host (e.g. localhost)
                upload_url = urlunparse(parsed_loc._replace(netloc=f"{webhdfs_host}:{parsed_loc.port or 9864}"))
                put_resp = requests.put(
                    upload_url,
                    data=payload,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=20,
                )
                if put_resp.status_code in (200, 201):
                    return
                last_error = f"DataNode HTTP {put_resp.status_code}: {put_resp.text[:250]}"
            elif init_resp.status_code in (200, 201):
                return
            else:
                last_error = f"NameNode HTTP {init_resp.status_code}: {init_resp.text[:250]}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(delay * (attempt + 1))
    raise RuntimeError(f"HDFS write failed for {path}. Last error: {last_error}")

