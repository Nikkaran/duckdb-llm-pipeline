# DuckDB + LLM Data Pipeline (AI-Native ETL)

An independent implementation of an AI-native ETL pipeline, adapting the concepts from the VLDB 2025 paper **"Beyond Quacking: Deep Integration of Language Models and RAG into DuckDB"** (arXiv:2504.01157).

🔗 **[View the Live Interactive Dashboard](https://duckdb-llm-pipeline.streamlit.app/)**

## 📖 Overview
This project demonstrates how to unify structured SQL analytics with unstructured text processing. It reads a dataset of customer reviews, uses **DuckDB** to perform relational aggregations (like product-level mean ratings), and uses a **Large Language Model** to perform semantic sentiment classification, merging the results back at the review level.

## 🏗️ Architecture & Engineering Choices
*   **Relational Engine (DuckDB):** Used as the in-process OLAP engine to scan the CSV and compute window functions efficiently on constrained hardware.
*   **Concurrency (Python `ThreadPoolExecutor`):** Because native DuckDB LLM extensions do not yet support all API providers (like DeepSeek) on Windows, I engineered a custom concurrent API wrapper. It uses thread-pooling to process reviews in parallel, alongside exponential backoff and retry logic for rate limits.
*   **Frontend (Streamlit + Custom CSS):** The UI is heavily customized to look like a vintage academic ledger, proving that data pipelines don't have to be visually boring.

## 🚀 Reproducibility (Quick Start)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the real Amazon reviews dataset (300 rows)
python prepare_data.py

# 3. Launch the dashboard
streamlit run app.py