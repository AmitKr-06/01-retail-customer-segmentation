# 🛍️ Customer Segmentation using RFM Analysis & Unsupervised Learning

## 📌 Overview
This project segments retail customers into distinct behavioral groups using **RFM (Recency, Frequency, Monetary) analysis** combined with **unsupervised machine learning**. The goal is to identify actionable customer segments — such as loyal high-value customers vs. at-risk/churned customers — to support data-driven marketing decisions.

The project covers the full ML lifecycle: data cleaning → feature engineering → clustering → algorithm comparison → production-ready code → REST API deployment.

---

## 📊 Dataset
- **Source**: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- **Original size**: 525,461 transactions
- **Cleaned size**: 399,568 transactions across 4,285 unique customers
- **Time period**: December 2009 – December 2010 (UK-based online retailer)

---

## 🧹 Data Cleaning
- Removed transactions with missing Customer ID (required to segment by customer)
- Removed cancelled orders (invoices prefixed with `'C'`)
- Removed non-product administrative entries (postage, manual adjustments, bank charges, discounts, test entries)
- Removed duplicate rows
- Removed transactions with zero/negative Quantity or Price
- **Result**: ~24% of raw rows removed as noise or non-purchase entries

---

## 🛠️ Feature Engineering — RFM

For each customer, three features were calculated from their full transaction history:

| Feature | Meaning | Example |
|---|---|---|
| **Recency** | Days since the customer's last purchase (lower = more recently active) | `10` = bought 10 days ago |
| **Frequency** | Total number of separate purchases made (higher = more loyal buyer) | `15` = shopped 15 times |
| **Monetary** | Total amount spent across all purchases (higher = more valuable customer) | `8000` = spent £8,000 total |

Applied a **log transform** to reduce skew, then standardized all features using `StandardScaler` before clustering — necessary because raw RFM values are heavily right-skewed and distance-based algorithms like KMeans are sensitive to scale.

---

## 🤖 Modeling & Algorithm Comparison

Three unsupervised clustering approaches were trained and compared:

| Algorithm | Approach | Silhouette Score | Notes |
|---|---|---|---|
| **KMeans (K=4)** | Centroid-based | **0.333** ✅ | Best separation, most interpretable |
| Hierarchical (Agglomerative) | Tree-based | 0.269 | Similar structure, weaker separation |
| DBSCAN | Density-based | 0.271 | Automatically flagged 54 customers as outliers |

Optimal K was selected using the **Elbow Method** and **Silhouette Score**, balanced against business interpretability — K=2 scored highest mathematically but was too simplistic to be actionable for a marketing team.

**KMeans was selected as the final model** for its stronger cluster separation and clearer business interpretation.

---

## 🎯 Final Customer Segments (KMeans, K=4)

| Segment | Recency | Frequency | Monetary | Count | Description |
|---|---|---|---|---|---|
| **Champions** | 13.8 days | 13.5 | £7,309 | 765 | Most recent, frequent, highest spend — your best customers |
| **Loyal** | 82.2 days | 4.1 | £1,787 | 1,174 | Steady engagement, good value |
| **New/Promising** | 22.3 days | 2.1 | £560 | 938 | Recently active, still building purchase history |
| **At Risk** | 184.7 days | 1.3 | £302 | 1,408 | Long inactive, low engagement — needs re-engagement |

---

## 🚀 API Usage

The trained model is served via a **FastAPI** REST endpoint. Send a customer's RFM values and receive their predicted segment.

### Example — a loyal, high-value customer
**Request:**
```json
POST /predict
{
  "recency": 10,
  "frequency": 15,
  "monetary": 8000
}
```
**Response:**
```json
{
  "cluster": 3,
  "segment": "Champions"
}
```

### Example — an inactive, low-value customer
**Request:**
```json
POST /predict
{
  "recency": 300,
  "frequency": 1,
  "monetary": 100
}
```
**Response:**
```json
{
  "cluster": 2,
  "segment": "At Risk"
}
```

---

## 🧰 Tech Stack
- **Language**: Python 3.13
- **Data Analysis**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (KMeans, DBSCAN, Agglomerative Clustering)
- **Visualization**: Matplotlib, Seaborn
- **API**: FastAPI, Uvicorn, Pydantic
- **Testing**: Pytest
- **Deployment**: Docker
- **CI/CD**: GitHub Actions

---

## 📂 Project Structure

├── .github/workflows/ # CI/CD pipeline (GitHub Actions)
├── api/ # FastAPI application
│ ├── main.py # API endpoints
│ └── schemas.py # Request/response validation
├── artifacts/ # Generated model, scaler, and processed data
├── data/ # Raw dataset
├── docker/ # Containerization files
├── models/ # (Alternate) saved model artifacts
├── notebooks/ # EDA and experimentation
├── reports/figures/ # Saved visualizations
├── src/
│ ├── components/ # Data ingestion, transformation, model training, evaluation
│ ├── pipeline/ # Training and prediction pipelines
│ ├── exception.py # Custom exception handling
│ ├── logger.py # Logging configuration
│ └── utils.py # Shared utility functions
├── tests/ # Unit tests
├── requirements.txt
└── README.md




---

## ⚙️ How to Run Locally

```bash
# Clone repo
git clone https://github.com/AmitKr-06/01-retail-customer-segmentation.git
cd 01-retail-customer-segmentation

# Setup environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run the full training pipeline (cleans data, trains model, evaluates)
python -m src.pipeline.training_pipeline

# Run the API
uvicorn api.main:app --reload
```
Then visit `http://127.0.0.1:8000/docs` for the interactive API documentation.

### Run Tests
```bash
pytest tests/ -v
```

### Run with Docker
```bash
docker compose -f docker/docker-compose.yml up --build
```

---

## 📈 Future Improvements
- Dynamic cluster-to-segment label mapping (currently hardcoded based on observed cluster centers)
- Customer Lifetime Value (CLV) prediction
- Cloud deployment (AWS/GCP/Azure)
- Batch prediction endpoint for scoring entire customer lists at once

---

## 👤 Author
**Amit Kumar**
[GitHub](https://github.com/AmitKr-06)