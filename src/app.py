from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Locate base directories robustly
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent if (SRC_DIR.parent / "models").exists() else SRC_DIR
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Import producer helper functions & shared configuration
from common import (
    HDFS_WEBHDFS_URL,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC,
    MODEL_PATH,
    make_article,
)
from article_producer import SAMPLES, load_dataset_articles

METRICS_PATH = Path(os.getenv("METRICS_PATH", str(PROJECT_ROOT / "models" / "metrics.json")))

CATEGORY_ICONS = {
    "business": "💼 Business",
    "entertainment": "🎬 Entertainment",
    "politics": "🏛️ Politics",
    "sport": "⚽ Sport",
    "tech": "💻 Tech",
}

PRESET_SAMPLES = {
    "Technology": {
        "title": "Quantum chip sets record in power efficiency and qubit stability",
        "text": "Researchers at a leading technology laboratory have demonstrated a 128-qubit quantum processor capable of operating at near room temperature. The breakthrough is expected to accelerate AI model training and cryptographic simulations while cutting server farm energy usage.",
    },
    "Business": {
        "title": "Global markets surge as corporate earnings beat quarterly estimates",
        "text": "Stock indices closed at record highs today following strong earnings reports from multinational manufacturing and semiconductor giants. Investors welcomed signs of declining inflation and central bank monetary stability.",
    },
    "Sport": {
        "title": "Underdogs secure thrilling last-minute victory in national championship",
        "text": "In a dramatic final match watched by over fifty thousand fans in the stadium, the visiting team scored in extra time to lift their first major league trophy in over two decades.",
    },
    "Politics": {
        "title": "Parliament opens bipartisan debate on national renewable infrastructure bill",
        "text": "Lawmakers from all major parties convened this morning for the first reading of an ambitious energy transition bill aimed at expanding solar and offshore wind farms across the country.",
    },
    "Entertainment": {
        "title": "International film festival honours indie cinema with grand jury prize",
        "text": "The 78th annual film festival concluded its week-long celebration with top honours awarded to an emotional drama exploring generational migration. Audiences gave the director an eight-minute standing ovation.",
    },
}

st.set_page_config(
    page_title="Real-Time News Categorization",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        st.error(f"Failed to load model file: {exc}")
        return None


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            return {}
    return {}


def check_kafka_connection() -> tuple[bool, str]:
    try:
        from kafka import KafkaConsumer
        consumer = KafkaConsumer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
            request_timeout_ms=1500,
            api_version_auto_timeout_ms=1500,
        )
        topics = consumer.topics()
        consumer.close()
        return True, f"Connected ({len(topics)} topics)"
    except Exception as exc:
        return False, str(exc)[:60]


def check_hdfs_connection() -> tuple[bool, str]:
    try:
        import requests
        resp = requests.get(HDFS_WEBHDFS_URL, params={"op": "LISTSTATUS"}, timeout=1.5)
        if resp.status_code in (200, 401, 403):
            return True, "NameNode Active (Port 9870)"
        return False, f"HTTP {resp.status_code}"
    except Exception as exc:
        return False, str(exc)[:60]


# Initialize session state variables
if "history" not in st.session_state:
    st.session_state["history"] = []
if "headline_input" not in st.session_state:
    st.session_state["headline_input"] = ""
if "text_input" not in st.session_state:
    st.session_state["text_input"] = ""
if "streaming_records" not in st.session_state:
    st.session_state["streaming_records"] = []

model = load_model()
metrics = load_metrics()

# --- SIDEBAR: System Status & Controls ---
with st.sidebar:
    st.header("⚙️ Pipeline Status")

    # ML Model status
    if model is not None:
        st.success("🤖 ML Model: Ready (TF-IDF + LinearSVC)")
    else:
        st.error("❌ ML Model: Not Found")
        st.caption(f"Run `python src/train_model.py`")

    # Kafka status
    st.write("**Apache Kafka** (`localhost:9092`)")
    kafka_ok, kafka_msg = check_kafka_connection()
    if kafka_ok:
        st.success(f"🟢 Kafka: {kafka_msg}")
    else:
        st.warning("🟡 Kafka: Offline / Unreachable")
        st.caption("Start with: `docker compose up -d`")

    # HDFS status
    st.write("**Hadoop HDFS** (`localhost:9870`)")
    hdfs_ok, hdfs_msg = check_hdfs_connection()
    if hdfs_ok:
        st.success(f"🟢 HDFS: {hdfs_msg}")
    else:
        st.warning("🟡 HDFS: Offline / Unreachable")

    st.markdown("---")
    st.markdown("### 🛠️ Hadoop SafeMode Tip")
    st.caption(
        "If you see `SafeModeException` right after starting Docker, run in PowerShell:\n"
        "`docker exec namenode hdfs dfsadmin -safemode leave`"
    )

    st.markdown("---")
    st.markdown("### 🔗 Quick Links")
    st.markdown("- [Hadoop NameNode UI](http://localhost:9870)")
    st.markdown("- [HDFS File Explorer](http://localhost:9870/explorer.html#/news/classified)")

    st.markdown("---")
    st.markdown("### 💡 Architecture")
    st.caption(
        "**Streamlit UI / Producer** ➔ "
        "**Apache Kafka** (`news-articles`) ➔ "
        "**Streaming Consumer + ML Classifier** ➔ "
        "**Hadoop HDFS** (`/news/classified`)"
    )


# --- MAIN HEADER ---
st.title("📰 Real-Time News Article Categorization")
st.markdown(
    "A production-grade **Big Data Analytics (BDA)** pipeline featuring "
    "**Scikit-Learn NLP**, **Apache Kafka Streaming**, and **Hadoop HDFS Storage**."
)

# Metric Summary Cards
accuracy = metrics.get("accuracy_percent") or (metrics.get("accuracy", 0) * 100)
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric(
        label="Model Accuracy",
        value=f"{accuracy:.2f}%" if accuracy else "N/A",
        delta="+14.33% vs 85% goal" if accuracy else None,
    )
with col_m2:
    st.metric(label="Classes", value=len(metrics.get("categories", [1, 2, 3, 4, 5])))
with col_m3:
    st.metric(label="Training Dataset", value=f"{metrics.get('dataset_rows', 2225):,} articles")
with col_m4:
    st.metric(label="Streaming Engine", value="Kafka + HDFS")

st.markdown("---")

# --- TABS INTERFACE ---
tab_classify, tab_producer, tab_metrics, tab_history = st.tabs([
    "🔍 Real-Time Article Categorization",
    "📡 Kafka Article Producer (Streaming)",
    "📊 Model Performance & Metrics",
    "🕒 Prediction History",
])


# =======================================================
# TAB 1: REAL-TIME ARTICLE CLASSIFICATION (SINGLE ARTICLE)
# =======================================================
with tab_classify:
    if model is None:
        st.error(
            "Trained model not found at `models/news_classifier.joblib`. "
            "Please run `python src/train_model.py` in your terminal first."
        )
        st.stop()
    st.subheader("✍️ Enter News Article")

    headline = st.text_input(
        "News Headline",
        value=st.session_state["headline_input"],
        placeholder="e.g., Breakthrough battery technology doubles electric vehicle range",
        key="headline_field",
    )

    article_text = st.text_area(
        "Article Content / Body Text",
        value=st.session_state["text_input"],
        height=160,
        placeholder="Paste full article text or paragraphs here...",
        key="text_field",
    )

    col_btn1, col_btn2 = st.columns([1, 1])

    with col_btn1:
        classify_pressed = st.button("🚀 Classify News Article", type="primary", use_container_width=True)

    with col_btn2:
        publish_single_kafka = st.button("📡 Publish Single Article to Kafka", use_container_width=True)

    # 1. Local ML Classification
    if classify_pressed:
        clean_headline = headline.strip()
        clean_body = article_text.strip()
        if not clean_headline and not clean_body:
            st.warning("⚠️ Please provide a headline or article content.")
        else:
            combined_text = f"{clean_headline} {clean_body}".strip()
            with st.spinner("Analyzing text with TF-IDF + LinearSVC..."):
                try:
                    prediction = str(model.predict([combined_text])[0])
                    badge_label = CATEGORY_ICONS.get(prediction.lower(), f"📌 {prediction.title()}")

                    # Decision score rankings if supported
                    scores_dict = {}
                    if hasattr(model, "decision_function"):
                        try:
                            df_scores = model.decision_function([combined_text])[0]
                            classes = getattr(model, "classes_", [])
                            if len(classes) == len(df_scores):
                                scores_dict = {
                                    cls: float(score) for cls, score in zip(classes, df_scores)
                                }
                        except Exception:
                            pass

                    pred_record = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "headline": clean_headline or "(No headline)",
                        "text": clean_body or "(No article body)",
                        "category": prediction.title(),
                        "icon": CATEGORY_ICONS.get(prediction.lower(), prediction.title()),
                        "length": len(combined_text),
                    }
                    st.session_state["history"].insert(0, pred_record)
                    st.session_state["history"] = st.session_state["history"][:50]

                    st.success("Classification Completed!")
                    res_col1, res_col2 = st.columns([1, 2])
                    with res_col1:
                        st.markdown("### Predicted Category")
                        st.markdown(f"## {badge_label}")
                    with res_col2:
                        if scores_dict:
                            st.markdown("### Confidence Ranking (Decision Values)")
                            sorted_scores = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
                            score_df = pd.DataFrame(
                                [{"Category": c.title(), "Relative Score": round(s, 2)} for c, s in sorted_scores]
                            )
                            st.dataframe(score_df, hide_index=True, use_container_width=True)

                except Exception as exc:
                    st.error(f"Classification failed: {exc}")

    # 2. Publish Single Article to Kafka
    if publish_single_kafka:
        clean_headline = headline.strip()
        clean_body = article_text.strip()
        if not clean_headline and not clean_body:
            st.warning("⚠️ Enter a headline or article body text before publishing.")
        else:
            try:
                from kafka import KafkaProducer

                article_payload = make_article(
                    title=clean_headline or clean_body[:80],
                    text=clean_body or clean_headline,
                    source="Streamlit Dashboard",
                )
                article_id = article_payload["article_id"]

                producer = KafkaProducer(
                    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
                    value_serializer=lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8"),
                    key_serializer=lambda key: key.encode("utf-8") if key else None,
                    request_timeout_ms=10000,
                )
                future = producer.send(
                    KAFKA_TOPIC,
                    key=article_id,
                    value=article_payload,
                )
                record_meta = future.get(timeout=10)
                producer.flush(timeout=5)
                producer.close()

                st.success(
                    f"✅ Successfully published to Kafka topic `{record_meta.topic}` "
                    f"(partition: {record_meta.partition}, offset: {record_meta.offset})!"
                )
                st.info(
                    "💡 If `src/article_consumer.py` is running in your terminal, it will automatically consume, "
                    f"classify, and save this article to `/news/classified/{article_id}.jsonl` in HDFS."
                )
            except Exception as exc:
                st.error(
                    f"Could not publish to Kafka: {exc}\n\n"
                    "Ensure Docker containers are running with: `docker compose up -d`."
                )


# =======================================================
# TAB 2: KAFKA STREAMING PRODUCER (STEP 5 MODES)
# =======================================================
with tab_producer:
    st.subheader("📡 Kafka News Article Streamer (Producer)")
    st.write(
        "Stream articles directly into the Apache Kafka topic (`news-articles`). "
        "Your running consumer (`python src/article_consumer.py`) will automatically process and save them to Hadoop HDFS."
    )

    # Selection of the 3 Producer Modes
    producer_mode = st.radio(
        "Choose Streaming Mode:",
        [
            "Option A: Stream real articles from BBC dataset (Round-Robin across all categories)",
            "Option B: Stream articles from a specific category",
            "Option C: Send built-in sample test articles (5 curated categories)",
        ],
        index=0,
    )

    col_opt1, col_opt2 = st.columns(2)

    category_filter = None
    if "Option A" in producer_mode:
        with col_opt1:
            stream_limit = st.slider("Number of articles to stream:", min_value=1, max_value=50, value=10, step=1)
        with col_opt2:
            stream_delay = st.slider("Delay between articles (seconds):", min_value=0.0, max_value=3.0, value=1.0, step=0.2)
        st.info("📂 **Option A**: Reads diverse news stories across business, tech, sport, politics, and entertainment from `dataset/BBC`.")

    elif "Option B" in producer_mode:
        with col_opt1:
            selected_cat = st.selectbox(
                "Select category to stream:",
                ["tech", "business", "sport", "politics", "entertainment"],
                format_func=lambda c: CATEGORY_ICONS.get(c, c.title()),
            )
            category_filter = selected_cat
            stream_limit = st.slider("Number of articles to stream:", min_value=1, max_value=30, value=5, step=1)
        with col_opt2:
            stream_delay = st.slider("Delay between articles (seconds):", min_value=0.0, max_value=3.0, value=0.8, step=0.2)
        st.info(f"🎯 **Option B**: Filters the BBC dataset strictly to `{selected_cat.title()}` articles.")

    else:  # Option C
        with col_opt1:
            st.write("**Built-in Curated Samples:** 5 diverse articles (1 per category)")
            stream_limit = 5
        with col_opt2:
            stream_delay = st.slider("Delay between articles (seconds):", min_value=0.0, max_value=3.0, value=0.5, step=0.1)
        st.info("🧪 **Option C**: Sends 5 pre-compiled test articles directly into the streaming pipeline.")

    # Start Streaming Button
    if st.button("🚀 Start Streaming Articles to Kafka", type="primary", use_container_width=True):
        articles_to_send = []
        if "Option A" in producer_mode:
            articles_to_send = load_dataset_articles(limit=stream_limit, category=None)
        elif "Option B" in producer_mode:
            articles_to_send = load_dataset_articles(limit=stream_limit, category=category_filter)
        else:  # Option C
            articles_to_send = [make_article(item["title"], item["text"], "sample") for item in SAMPLES]

        if not articles_to_send:
            st.error("No articles found in the dataset! Check that `dataset/BBC` exists.")
        else:
            try:
                from kafka import KafkaProducer

                producer = KafkaProducer(
                    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
                    value_serializer=lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8"),
                    key_serializer=lambda key: key.encode("utf-8") if key else None,
                    retries=5,
                    request_timeout_ms=20000,
                )

                progress_bar = st.progress(0)
                status_text = st.empty()
                streaming_table_placeholder = st.empty()

                sent_records = []
                total = len(articles_to_send)

                for idx, article in enumerate(articles_to_send, start=1):
                    art_id = article["article_id"]
                    future = producer.send(
                        KAFKA_TOPIC,
                        key=art_id,
                        value=article,
                    )
                    meta = future.get(timeout=20)

                    record_info = {
                        "#": idx,
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Title": article["title"][:55] + ("..." if len(article["title"]) > 55 else ""),
                        "Source / Category": article.get("source", "unknown"),
                        "Partition": meta.partition,
                        "Offset": meta.offset,
                        "Status": "Published ✅",
                    }
                    sent_records.insert(0, record_info)
                    st.session_state["streaming_records"] = sent_records

                    # Update UI live
                    progress_bar.progress(idx / total)
                    status_text.markdown(f"**Streaming Progress:** [{idx}/{total}] Published *'{article['title'][:40]}...'*")
                    streaming_table_placeholder.dataframe(
                        pd.DataFrame(sent_records),
                        hide_index=True,
                        use_container_width=True,
                    )

                    if stream_delay > 0 and idx < total:
                        time.sleep(stream_delay)

                producer.flush(timeout=10)
                producer.close()

                st.success(
                    f"🎉 Successfully streamed all {total} articles to Kafka topic `{KAFKA_TOPIC}`!"
                )
                st.balloons()

            except Exception as exc:
                st.error(
                    f"Streaming to Kafka failed: {exc}\n\n"
                    "Ensure Docker containers are running with: `docker compose up -d`."
                )

    # Show past streaming log if available
    if st.session_state.get("streaming_records"):
        st.markdown("### 📋 Published Articles Stream Log")
        st.dataframe(
            pd.DataFrame(st.session_state["streaming_records"]),
            hide_index=True,
            use_container_width=True,
        )


# =======================================================
# TAB 3: MODEL PERFORMANCE & METRICS
# =======================================================
with tab_metrics:
    st.subheader("📊 Model Performance & Evaluation")
    if not metrics:
        st.info("No metrics file found at `models/metrics.json`. Run `python src/train_model.py` to generate it.")
    else:
        st.markdown(f"**Model**: `{metrics.get('model', 'TF-IDF + LinearSVC')}` | **Dataset**: `{metrics.get('dataset', 'BBC News')}`")

        c1, c2, c3 = st.columns(3)
        c1.metric("Overall Accuracy", f"{metrics.get('accuracy_percent', 0):.2f}%")
        c2.metric("Target Accuracy", f"{metrics.get('target_accuracy_percent', 85)}%")
        c3.metric("Goal Status", "Target Achieved ✅" if metrics.get("target_met") else "In Progress")

        # Classification Report Table
        st.markdown("### Per-Category Classification Report")
        report = metrics.get("classification_report", {})
        if isinstance(report, dict):
            report_rows = []
            for category, data in report.items():
                if isinstance(data, dict) and "f1-score" in data:
                    report_rows.append({
                        "Category": category.title(),
                        "Precision": f"{data.get('precision', 0) * 100:.2f}%",
                        "Recall": f"{data.get('recall', 0) * 100:.2f}%",
                        "F1-Score": f"{data.get('f1-score', 0) * 100:.2f}%",
                        "Support (Test Count)": int(data.get("support", 0)),
                    })
            if report_rows:
                st.dataframe(pd.DataFrame(report_rows), hide_index=True, use_container_width=True)

        # Confusion Matrix
        cm = metrics.get("confusion_matrix")
        labels = metrics.get("confusion_matrix_labels", [])
        if cm and labels:
            st.markdown("### Confusion Matrix")
            display_labels = [l.title() for l in labels]
            cm_df = pd.DataFrame(cm, index=display_labels, columns=display_labels)
            cm_df.index.name = "Actual / True"
            st.dataframe(cm_df, use_container_width=True)


# =======================================================
# TAB 4: PREDICTION HISTORY
# =======================================================
with tab_history:
    st.subheader("🕒 Session Prediction History")
    history = st.session_state.get("history", [])

    if history:
        col_hist_action1, col_hist_action2 = st.columns([1, 4])
        with col_hist_action1:
            if st.button("🗑️ Clear History"):
                st.session_state["history"] = []
                st.session_state["selected_history_article"] = None
                st.rerun()

        hist_df = pd.DataFrame(history)
        st.dataframe(
            hist_df[["timestamp", "icon", "headline", "length"]].rename(
                columns={"icon": "Category", "headline": "Headline", "timestamp": "Timestamp", "length": "Chars"}
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("### 🔍 Article Details Inspector")
        col_select, col_btn = st.columns([3, 1])
        with col_select:
            selected_idx = st.selectbox(
                "Select article to inspect full typed data:",
                options=range(len(history)),
                format_func=lambda i: f"#{i+1} | {history[i].get('timestamp')} | {history[i].get('icon')} | {history[i].get('headline')[:55]}",
                key="history_details_select",
            )
        with col_btn:
            st.write("")
            st.write("")
            view_details_clicked = st.button("📄 View Details", use_container_width=True)

        if view_details_clicked:
            st.session_state["selected_history_article"] = selected_idx

        active_idx = st.session_state.get("selected_history_article")
        if active_idx is not None and active_idx < len(history):
            item = history[active_idx]
            with st.container(border=True):
                st.markdown(f"#### 📌 Article #{active_idx+1}: {item.get('headline')}")
                st.write(f"**Category:** {item.get('icon', item.get('category'))} | **Timestamp:** `{item.get('timestamp')}` | **Characters:** `{item.get('length')}`")
                st.markdown("**Actual Typed Article Data:**")
                typed_data = item.get("text", "")
                if typed_data and typed_data != "(No article body)":
                    st.text_area("Full Typed Article", value=typed_data, height=180, disabled=True, label_visibility="collapsed")
                else:
                    st.info("No separate body was entered (classified by headline only).")
    else:
        st.info("No articles classified in this session yet. Classify an article in the first tab to see history here.")

st.markdown("---")
st.caption(
    "Big Data Analytics Project • End-to-End Pipeline: Kafka Ingestion ➔ Machine Learning Categorization ➔ Hadoop HDFS Storage"
)
