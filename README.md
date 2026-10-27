# Real-Time News Article Categorization

A production-grade Big Data Analytics (BDA) pipeline that implements real-time text classification using **Apache Kafka**, **Hadoop HDFS**, and **Scikit-Learn NLP**:
- **Dataset**: Automatically loads and compiles the 5-class BBC News dataset (2,225 articles across `business`, `entertainment`, `politics`, `sport`, `tech`) from the local `dataset/BBC` folder.
- **Machine Learning**: Trains a high-performance **TF-IDF + LinearSVC** classifier achieving **~99.3% accuracy** on held-out test data.
- **Real-Time Streaming**: Streams live and dataset news articles into an **Apache Kafka** topic (`news-articles`).
- **Distributed Ingestion & Classification**: A streaming consumer ingests messages from Kafka, classifies articles in real time, and persists raw and categorized JSONL records to **Hadoop HDFS** via WebHDFS.

---

## Architecture Flow

```
[ dataset/BBC / Custom Articles ]
               │
               ▼
   `src/article_producer.py`
               │  (Publishes JSON payloads)
               ▼
      Apache Kafka Broker
   Topic: `news-articles` (Port 9092)
               │
               ▼
   `src/article_consumer.py`
               │  (TF-IDF + LinearSVC Inference)
               ▼
      Hadoop Distributed File System (HDFS)
      ├── /news/raw/<article_id>.jsonl
      └── /news/classified/<article_id>.jsonl
```

---

## Prerequisites

- **OS**: Windows 10/11 (or Linux / macOS)
- **Docker Desktop**: Docker Compose v2 enabled and running
- **Python**: Version 3.10+ installed and added to `PATH`
- **RAM**: Minimum 4 GB free RAM for Kafka and Hadoop containers

---

## Step-by-Step Guide to Run the Project

### Step 1: Start Docker Infrastructure (Kafka & Hadoop)

Open **PowerShell** in the project root directory:

```powershell
docker compose up -d
docker compose ps
```

Ensure all three containers (`kafka`, `namenode`, `datanode`) show `STATUS: Up (healthy)`.
> *Wait approximately 30–60 seconds for Hadoop NameNode and DataNode to complete startup health checks.*

You can also monitor the Hadoop NameNode web interface in your browser:
- **Hadoop NameNode UI**: [http://localhost:9870](http://localhost:9870)

---

### Step 2: Set Up Python Virtual Environment

In the same PowerShell terminal, create and activate a virtual environment, then install dependencies:

```powershell
# Create virtual environment
py -3.10 -m venv .venv

# If script execution is restricted on Windows, enable it for this process:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install requirements
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 3: Train and Evaluate the NLP Model

The BBC dataset is located in `dataset/BBC` with folders for each category (`business`, `entertainment`, `politics`, `sport`, `tech`).

Run the training script:

```powershell
python src/train_model.py
```

**What this does:**
1. Automatically discovers `dataset/BBC` (or `dataset/BCC`), reads all 2,225 text articles, and compiles `data/bbc-text.csv`.
2. Splits into stratified train (80%) and test (20%) sets.
3. Fits an n-gram TF-IDF vectorizer and LinearSVC model.
4. Outputs classification metrics (Precision, Recall, F1-Score, Confusion Matrix).
5. Saves the trained model to `models/news_classifier.joblib` and metrics to `models/metrics.json`.

*(Held-out test accuracy is typically **>99%**, significantly beating the 85% project baseline).*

---

### Step 4: Start the Real-Time Streaming Consumer

Open a **second PowerShell window**, navigate to the project directory, activate `.venv`, and start the consumer:

```powershell
.\.venv\Scripts\Activate.ps1
python src/article_consumer.py
```

The consumer will connect to Kafka broker (`localhost:9092`), subscribe to the `news-articles` topic, ensure the HDFS directories `/news/raw` and `/news/classified` exist, and wait for incoming articles.

Keep this terminal window running to observe predictions live.

---

### Step 5: Publish News Articles (Producer)

You can publish articles using either the **interactive Streamlit Web Dashboard (Tab 2: Kafka Article Producer)** or via the PowerShell CLI:

#### Option A: Stream real articles directly from the `dataset/BBC` folder (Recommended)
Stream 10 diverse articles with a 1-second delay between each:
```powershell
.\.venv\Scripts\Activate.ps1
python src/article_producer.py --dataset --limit 10 --delay 1.0
```

#### Option B: Stream articles from a specific category
Filter streaming to only articles from the `tech` category:
```powershell
python src/article_producer.py --dataset --category tech --limit 5 --delay 0.8
```

#### Option C: Send built-in sample test articles
Send 5 curated sample articles covering all categories:
```powershell
python src/article_producer.py --sample
```

#### Option D: Publish a custom article
Publish any custom title and text:
```powershell
python src/article_producer.py --title "AI Breakthrough in Quantum Computing" --text "Scientists developed a new quantum processor with error correction algorithms powered by machine learning."
```

As soon as articles are sent, switch to **Terminal 2 (Consumer)**: you will see real-time classification output with predicted categories and HDFS file paths!

---

### Step 6: Verify and Query Stored Articles in Hadoop HDFS

Verify that articles were stored in Hadoop HDFS using `docker exec`:

#### 1. List files in HDFS:
```powershell
docker exec namenode hdfs dfs -ls -R /news
```

#### 2. Inspect classified articles (JSONL):
```powershell
docker exec namenode hdfs dfs -cat /news/classified/*.jsonl
```

#### 3. View through Hadoop Web UI:
Open [http://localhost:9870/explorer.html#/news/classified](http://localhost:9870/explorer.html#/news/classified) in your browser to inspect and download individual records directly from the NameNode file explorer.

Each record stored in HDFS contains:
```json
{
  "article_id": "c701d1d2-c41d-42c4-829e-4b29bc5ac2cc",
  "title": "AI Breakthrough in Quantum Computing",
  "text": "Scientists developed a new quantum processor...",
  "source": "manual",
  "published_at": "2026-10-15T08:48:31.979603+00:00",
  "kafka_topic": "news-articles",
  "kafka_partition": 0,
  "kafka_offset": 10,
  "received_at": "2026-10-15T08:48:32.062098+00:00",
  "predicted_category": "tech",
  "model_name": "TF-IDF + LinearSVC",
  "classified_at": "2026-10-15T08:48:32.064500+00:00"
}
```

---

### Step 7: Launch the Streamlit Frontend Web App

Start the interactive web dashboard to test classifications, inspect real-time metrics, and publish articles to Kafka:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run src/app.py
```

The dashboard will open automatically in your browser at **http://localhost:8501**.

Features available in the Streamlit UI:
- **Instant Classification**: Enter any headline and text (or click one of the preset sample buttons) to classify with TF-IDF + LinearSVC.
- **Kafka Publishing**: Send news articles directly into the streaming pipeline (`news-articles` topic) to watch them flow into HDFS.
- **Performance & Metrics**: View model accuracy (99.33%), classification report per class, and confusion matrix.
- **Prediction History**: Track all classified articles and categories in your current session.
- **Pipeline Health**: Check Kafka broker and Hadoop WebHDFS connection status directly from the sidebar.

---

## Stopping and Restarting Services

- To stop the Python consumer: Press `Ctrl + C` in its terminal window.
- To temporarily stop Docker containers:
  ```powershell
  docker compose stop
  ```
- To restart stopped containers:
  ```powershell
  docker compose start
  ```
- To shut down and remove containers while preserving HDFS data volumes:
  ```powershell
  docker compose down
  ```
- To reset all data and start completely fresh:
  ```powershell
  docker compose down -v
  ```

---

## Troubleshooting & Notes

1. **Docker Container Name Conflicts**:
   If `docker compose up -d` reports a container name conflict (e.g. `The container name "/kafka" is already in use`), remove the old container:
   ```powershell
   docker rm -f kafka
   docker compose up -d
   ```

2. **WebHDFS Redirection on Windows**:
   When writing to Hadoop HDFS from the host, NameNode redirects client requests to DataNode (`port 9864`). The included WebHDFS client in `src/common.py` automatically remaps internal container hostnames to `localhost:9864` to prevent Windows DNS resolution errors.

3. **Execution Policy Error in PowerShell**:
   If activating `.venv` gives `cannot be loaded because running scripts is disabled on this system`, run:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```

4. **Dataset Location**:
   The loader automatically checks `dataset/BBC`, `dataset/BCC`, or `dataset/`. If you place text files inside category folders (`business`, `entertainment`, `politics`, `sport`, `tech`), `src/train_model.py` and `src/article_producer.py --dataset` will immediately detect and use them.

---

## Project Directory Structure

```
real_time_news_categorization/
├── data/
│   └── bbc-text.csv               # Compiled CSV dataset
├── dataset/
│   └── BBC/                       # Raw BBC text articles
│       ├── business/              # 510 articles (.txt)
│       ├── entertainment/         # 386 articles (.txt)
│       ├── politics/              # 417 articles (.txt)
│       ├── sport/                 # 511 articles (.txt)
│       └── tech/                  # 401 articles (.txt)
├── docker-compose.yml             # Kafka (9092) and Hadoop (9870, 9864) services
├── models/
│   ├── metrics.json               # Evaluation metrics & confusion matrix
│   └── news_classifier.joblib     # Serialized TF-IDF + LinearSVC model
├── requirements.txt               # Python package dependencies
├── src/
│   ├── article_consumer.py        # Kafka consumer + classifier + HDFS sink
│   ├── article_producer.py        # Kafka producer (sample, dataset, custom)
│   ├── common.py                  # WebHDFS client & shared configuration
│   └── train_model.py             # Model training & dataset compiler
└── README.md                      # Comprehensive execution guide
```
