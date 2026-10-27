from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.request import urlopen

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_PATH = DATA_DIR / "bbc-text.csv"
DATA_URL = "https://raw.githubusercontent.com/selva86/datasets/master/bbc-text.csv"


CATEGORIES = {"business", "entertainment", "politics", "sport", "tech"}


def find_dataset_dir() -> Path | None:
    candidate_paths = [
        ROOT / "dataset" / "BBC",
        ROOT / "dataset" / "BCC",
        ROOT / "dataset",
        ROOT / "data" / "BBC",
        ROOT / "data" / "BCC",
    ]
    for p in candidate_paths:
        if p.is_dir() and CATEGORIES.issubset({d.name.lower() for d in p.iterdir() if d.is_dir()}):
            return p
    dataset_root = ROOT / "dataset"
    if dataset_root.is_dir():
        for d in dataset_root.rglob("*"):
            if d.is_dir() and CATEGORIES.issubset({sub.name.lower() for sub in d.iterdir() if sub.is_dir()}):
                return d
    return None


def load_dataset() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = find_dataset_dir()
    if raw_dir:
        print(f"Loading BBC dataset from local directory: {raw_dir}")
        records = []
        for cat in sorted(CATEGORIES):
            cat_dir = next((d for d in raw_dir.iterdir() if d.is_dir() and d.name.lower() == cat), None)
            if not cat_dir:
                continue
            for file_path in sorted(cat_dir.glob("*.txt")):
                try:
                    text = file_path.read_text(encoding="utf-8", errors="replace").strip()
                except Exception:
                    text = file_path.read_text(encoding="latin-1", errors="replace").strip()
                if text:
                    records.append({"category": cat, "text": text})
        df = pd.DataFrame(records)
        if not df.empty:
            df.to_csv(DATA_PATH, index=False, encoding="utf-8")
            print(f"Saved compiled dataset to {DATA_PATH} ({len(df)} rows)")
    elif DATA_PATH.exists():
        print(f"Loading dataset from: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        print(f"Downloading BBC News dataset from: {DATA_URL}")
        try:
            with urlopen(DATA_URL, timeout=30) as response:
                DATA_PATH.write_bytes(response.read())
        except Exception as exc:
            raise RuntimeError(
                "Dataset download failed and local dataset folder not found. "
                "Ensure dataset is placed in 'dataset/BBC' or place 'data/bbc-text.csv'."
            ) from exc
        df = pd.read_csv(DATA_PATH)

    required = {"category", "text"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns {sorted(required)}; found {list(df.columns)}")
    df = df[["category", "text"]].dropna()
    df["category"] = df["category"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[(df["category"] != "") & (df["text"] != "")]
    labels = sorted(df["category"].unique().tolist())
    print(f"Dataset rows: {len(df)}")
    print(f"Categories ({len(labels)}): {', '.join(labels)}")
    if len(labels) != 5:
        raise ValueError(f"Expected exactly 5 categories, found {len(labels)}: {labels}")
    return df



def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["category"], test_size=0.20, random_state=42, stratify=df["category"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, stop_words="english", max_features=100_000,
            ngram_range=(1, 2), sublinear_tf=True
        )),
        ("classifier", LinearSVC(C=1.0, random_state=42)),
    ])
    print(f"\nTraining on {len(X_train)} articles...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    labels = sorted(df["category"].unique().tolist())
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    print(f"\nHeld-out test accuracy: {accuracy * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (alphabetical category order):")
    print(matrix)
    model_path = MODEL_DIR / "news_classifier.joblib"
    joblib.dump(model, model_path)
    metrics = {
        "dataset": "BBC News", "dataset_rows": int(len(df)), "categories": labels,
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "accuracy": accuracy, "accuracy_percent": round(accuracy * 100, 2),
        "target_accuracy_percent": 85, "target_met": bool(accuracy > 0.85),
        "classification_report": report, "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": labels, "model": "TF-IDF + LinearSVC",
        "random_state": 42
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {MODEL_DIR / 'metrics.json'}")
    if accuracy <= 0.85:
        print("WARNING: Accuracy did not exceed 85%; consider tuning the model.")
    else:
        print("Success: accuracy exceeded the 85% target.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
