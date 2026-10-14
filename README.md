# News Classification 📰

An end-to-end Natural Language Processing (NLP) pipeline designed for multi-class news article classification and automated topical routing.

> [!NOTE]
> **Learning Purpose**: This project was built strictly for **educational and learning purposes** as part of an applied NLP curriculum. It demonstrates practical, production-oriented patterns for data ingestion, text preprocessing, feature engineering, class rebalancing, multi-class classification, topical data routing, and REST API deployment.

---

## 📑 Table of Contents
- [Project Overview](#project-overview)
- [Architecture & NLP Pipeline](#architecture--nlp-pipeline)
  - [1. Data Ingestion](#1-data-ingestion)
  - [2. Preprocessing & Lemmatization (spaCy + Regex)](#2-preprocessing--lemmatization-spacy--regex)
  - [3. Feature Extraction (TF-IDF Vectorization)](#3-feature-extraction-tf-idf-vectorization)
  - [4. Class Balancing & Model Training (RandomOverSampler + LinearSVC)](#4-class-balancing--model-training-randomoversampler--linearsvc)
  - [5. Model Evaluation](#5-model-evaluation)
  - [6. Automated Feed Routing](#6-automated-feed-routing)
  - [7. API Serving (FastAPI)](#7-api-serving-fastapi)
- [Repository Structure](#repository-structure)
- [Categories Classified](#categories-classified)
- [Installation & Setup](#installation--setup)
- [Step-by-Step Execution Guide](#step-by-step-execution-guide)
- [API Usage](#api-usage)
- [License](#license)

---

## Project Overview

In digital media and news aggregation platforms, thousands of news articles are published across varying domains every hour. Manually triaging, tagging, and distributing articles to editorial desks or topic-specific feeds is slow and cost-prohibitive.

This project delivers an automated pipeline that:
1. Combines news headlines and article bodies for maximal semantic representation.
2. Cleans raw unstructured text using high-throughput vectorized operations and spaCy lemmatization.
3. Converts text into high-dimensional numerical feature spaces via TF-IDF (unigrams + bigrams).
4. Solves class distribution imbalance using `imblearn` oversampling strictly on the training set.
5. Trains an optimized Support Vector Machine (`LinearSVC`) for robust multi-class text categorization.
6. Evaluates model performance via precision, recall, F1-score, and a confusion matrix.
7. Automatically routes classified news items into dedicated category feed CSVs (`data/feeds/{category}.csv`).
8. Deploys the inference pipeline as a low-latency REST API using FastAPI.

---

## Architecture & NLP Pipeline

```
Raw Inshort Datasets (data/raw/inshort_news_data-1..7.csv)
       │
       ▼
[ Stage 1: Data Ingestion & Combination ]
       ├─ src/data/load_data.py
       └─ news_headline + news_article
       │
       ▼
[ Stage 2: Text Preprocessing & Cleaning ]
       ├─ Vectorized lowercasing & regex cleanup (URLs, non-alpha)
       ├─ spaCy (en_core_web_sm) Tokenization & Lemmatization (batch size 200)
       ├─ Stop-word removal with parser/ner disabled for speed
       └─ Output: data/processed/cleaned_news.csv
       │
       ▼
[ Stage 3: Feature Extraction (TF-IDF) ]
       ├─ src/features/vectorizer.py
       ├─ N-gram range: (1, 2) [Unigrams + Bigrams]
       ├─ Vocabulary cap: max_features = 20,000
       └─ Artifact: models/tfidf_vectorizer.pkl
       │
       ▼
[ Stage 4: Split & Class Balancing ]
       ├─ Stratified Train/Test Split (80/20)
       ├─ RandomOverSampler on training data only
       └─ LinearSVC(C=1.0) One-vs-Rest Classifier
       │
       ▼
[ Stage 5: Evaluation & Persistence ]
       ├─ Classification Report & Confusion Matrix
       └─ Artifact: models/news_model.pkl
       │
       ▼
[ Stage 6 & 7: Serving & Real-Time Feed Routing ]
       ├─ Interactive Terminal Predictor (src/models/predict.py)
       ├─ FastAPI Service (src/api/app.py: POST /predict)
       └─ Topical Dispatch: data/feeds/{category}.csv
```

---

### 1. Data Ingestion
- **Module**: `src/data/load_data.py`
- Combines multiple raw Inshorts news extracts (`inshort_news_data-1.csv` through `inshort_news_data-7.csv`).
- Concatenates the article headline and full body text to supply the downstream classifier with both high-level context and detailed keyword evidence.

### 2. Preprocessing & Lemmatization (spaCy + Regex)
- **Module**: `src/data/preprocess.py` & `src/data/process_dataset.py`
- **Vectorized String Sanitation**: Strips external hyperlinks (`http\S+`) and non-alphabetic characters using fast pandas regex before passing data to NLP models.
- **Batched spaCy Processing**: Leverages `nlp.pipe()` with a batch size of 200, disabling resource-intensive components (`parser`, `ner`) to optimize CPU throughput during tokenization and lemmatization.
- **Stop-word Filtering**: Eliminates noise words while keeping meaningful root tokens (`token.lemma_`).

### 3. Feature Extraction (TF-IDF Vectorization)
- **Module**: `src/features/vectorizer.py`
- Converts text documents into TF-IDF numerical matrices:
  - **N-Grams (`(1, 2)`)**: Preserves contextual phrases (e.g., *"artificial intelligence"*, *"stock market"*, *"premier league"*).
  - **Vocabulary Threshold (`max_features=20000`)**: Bounds memory consumption and curtails overfitting by retaining the 20,000 most informative n-grams.

### 4. Class Balancing & Model Training (RandomOverSampler + LinearSVC)
- **Module**: `src/models/build_model.py` & `src/models/train.py`
- **Leakage Prevention**: Stratified 80/20 train/test split is performed *before* any oversampling, ensuring test distributions reflect real-world evaluations.
- **Imbalance Handling**: Uses `RandomOverSampler` from `imblearn` on the training split to equalize sample counts across minority categories.
- **Linear Support Vector Classifier**: Uses `LinearSVC(C=1.0)`, well-suited for high-dimensional sparse text vectors due to its efficiency and margin maximization.

### 5. Model Evaluation
- **Module**: `src/models/evaluate.py`
- Computes precision, recall, and F1-score across all categories on the unseen test split.
- Generates a labeled confusion matrix cross-referencing actual vs. predicted classes.

### 6. Automated Feed Routing
- **Module**: `src/routing/store.py`
- On each classification, writes the incoming article, predicted category, and ISO timestamp to `data/feeds/{category}.csv`.
- Ensures automated logging and separation of articles for downstream consumption or analytics.

### 7. API Serving (FastAPI)
- **Module**: `src/api/app.py`
- Provides a lightweight HTTP endpoint (`POST /predict`) powered by FastAPI and Uvicorn.
- Automates preprocessing, inference, and real-time routing within a single request cycle.

---

## Repository Structure

```
news__classification/
│
├── data/
│   ├── raw/                  # 7 raw Inshorts news CSV files
│   ├── processed/            # Cleaned dataset (cleaned_news.csv)
│   └── feeds/                # Categorized articles (e.g., science.csv, world.csv)
│
├── models/
│   ├── tfidf_vectorizer.pkl  # Trained TF-IDF feature extractor
│   └── news_model.pkl        # Trained LinearSVC classification model
│
├── notebooks/                # Jupyter exploration / experimentation directory
│
├── src/
│   ├── api/
│   │   └── app.py            # FastAPI prediction & routing web service
│   ├── data/
│   │   ├── load_data.py      # Multi-file dataset loader
│   │   ├── preprocess.py     # Text cleaning & spaCy lemmatization helpers
│   │   └── process_dataset.py# End-to-end cleaning pipeline runner
│   ├── features/
│   │   └── vectorizer.py     # TF-IDF fitting and serialization
│   ├── models/
│   │   ├── build_model.py    # Classifier instantiation (LinearSVC)
│   │   ├── train.py          # Split, oversampling, and model training
│   │   ├── evaluate.py       # Metrics and confusion matrix reporting
│   │   └── predict.py        # Interactive CLI inference loop
│   ├── routing/
│   │   └── store.py          # Category-based feed dispatcher
│   └── utils/                # Helper utilities
│
├── main.py                   # Quick-start entry point for data processing
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
└── README.md                 # Project documentation
```

---

## Categories Classified

The model classifies news articles across 7 categories:

| Category | Example Topics |
| :--- | :--- |
| **`world`** | International diplomacy, global events, foreign policies |
| **`entertainment`** | Cinema, music, celebrity news, awards |
| **`sports`** | Cricket, football, tournaments, athletic achievements |
| **`technology`** | AI, software, hardware releases, cybersecurity |
| **`politics`** | Governance, elections, policy debates, legislation |
| **`science`** | Space exploration, medical research, environmental discoveries |
| **`automobile`** | Electric vehicles, car launches, transport industry |

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/teevvr7/news__classification.git
cd news__classification
```

### 2. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Step-by-Step Execution Guide

Run the pipeline sequentially from the root directory:

### Step 1: Preprocess the Raw Data
Combines the raw Inshorts datasets, cleans the text, applies spaCy lemmatization, and outputs `data/processed/cleaned_news.csv`:
```bash
python main.py
```

### Step 2: Extract TF-IDF Features
Fits the TF-IDF vectorizer and saves `models/tfidf_vectorizer.pkl`:
```bash
python -m src.features.vectorizer
```

### Step 3: Train the Classifier
Splits data (80/20), balances categories using `RandomOverSampler`, trains `LinearSVC`, and saves `models/news_model.pkl`:
```bash
python -m src.models.train
```

### Step 4: Evaluate the Model
Prints the classification report (precision, recall, f1-score) and confusion matrix on the test set:
```bash
python -m src.models.evaluate
```

### Step 5: Interactive Predictions (Terminal)
Launch the CLI loop to enter news text and receive real-time classification predictions:
```bash
python -m src.models.predict
```

### Step 6: Run the API Microservice
Start the FastAPI server using Uvicorn:
```bash
uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
```
Interactive Swagger API docs will be accessible at: `http://127.0.0.1:8000/docs`

---

## API Usage

### Endpoint: `/predict`
- **Method**: `POST`
- **Request Body**:
```json
{
  "text": "NASA's James Webb Space Telescope has discovered water vapor in the inner disk of a planetary system."
}
```
- **Response**:
```json
{
  "category": "science"
}
```
*Note: Each classified article is automatically routed and saved to `data/feeds/{category}.csv`.*

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for complete details.
