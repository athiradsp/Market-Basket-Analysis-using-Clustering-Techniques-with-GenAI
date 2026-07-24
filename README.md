# Market Basket Analysis using Clustering Techniques with GenAI

An end-to-end modular Python web application built with Streamlit for Market Basket Analysis, Customer Segmentation, Association Rule Mining, and Generative AI Business Intelligence.

---

## 🌟 Key Features

1. **Synthetic Data Generation & Data Cleaning (`data_loader.py`)**:
   - Generates realistic transactional datasets with co-purchased bundles.
   - Cleans missing `CustomerID`s, filters canceled invoices (`'C'`), non-positive quantities/prices.
   - Computes RFM metrics (Recency, Frequency, Monetary) and scales features with `StandardScaler`.
   - Constructs binary one-hot transaction matrices for itemset mining.

2. **Unsupervised Customer Segmentation (`clustering.py`)**:
   - **K-Means Clustering**: WCSS/Inertia calculation and Elbow curve visualization.
   - **DBSCAN Clustering**: Density-based cluster & noise identification.
   - **Agglomerative Hierarchical Clustering**: Linkage tree clustering.
   - Evaluation metrics: **Silhouette Score** $S(i)$ and **Davies-Bouldin Index** (DBI).

3. **Association Rule Mining (`association_rules.py`)**:
   - **Apriori Algorithm** vs **FP-Growth Algorithm** implementation.
   - Execution time benchmarking (Apriori vs FP-Growth speedup).
   - Core metrics: Support, Confidence, Lift.
   - Interactive 2D Network Graph visualizer using NetworkX & Plotly.

4. **GenAI Insight Generation Layer (`genai_insights.py`)**:
   - Multi-provider LLM support: **OpenAI**, **Anthropic**, **Ollama**, and **Offline Heuristic Synthesizer**.
   - Translates ML cluster centroids and high-lift rules into structured JSON prompts.
   - Generates Customer Personas, Cross-Selling Strategies, and Inventory Shelf Placement recommendations.

5. **Interactive Web Dashboard (`app.py`)**:
   - Multi-tab Streamlit dashboard with modern dark glassmorphism styling.
   - Tab 1: Dataset Overview & EDA
   - Tab 2: Customer Clustering
   - Tab 3: Association Rule Mining & Network Graph
   - Tab 4: GenAI BI Report & TXT Export
   - Tab 5: Technical Documentation & Mathematical Formula Guide

---

## 🚀 Installation & Running Locally

```bash
# 1. Clone or navigate to project directory
cd MLproject

# 2. Install dependencies for local Streamlit dashboard
pip install -r requirements-streamlit.txt

# 3. Launch Streamlit Dashboard
streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## 📖 Mathematical Formulas & Metrics Reference

- **Silhouette Score**:
  $$S(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
- **Davies-Bouldin Index**:
  $$DB = \frac{1}{k} \sum_{i=1}^k \max_{j \neq i} \left( \frac{\sigma_i + \sigma_j}{d(c_i, c_j)} \right)$$
- **Support**:
  $$\text{Support}(A \rightarrow B) = \frac{\text{Count}(A \cap B)}{N}$$
- **Confidence**:
  $$\text{Confidence}(A \rightarrow B) = \frac{\text{Support}(A \cap B)}{\text{Support}(A)}$$
- **Lift**:
  $$\text{Lift}(A \rightarrow B) = \frac{\text{Confidence}(A \rightarrow B)}{\text{Support}(B)}$$
