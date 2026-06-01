# Credit Scoring Model

An end-to-end credit risk probability model built for Bati Bank's buy-now-pay-later partnership with an eCommerce platform. This project transforms raw transaction data into a deployed model service that scores new applicants in real time.

![CI](https://github.com/RigbeWeleslasie/credit-scoring-model/actions/workflows/ci.yml/badge.svg)

---

## Project Structure

```
credit-scoring-model/
├── .github/workflows/ci.yml       # CI/CD: flake8 + pytest + Docker build
├── data/
│   ├── raw/                        # Raw data (gitignored)
│   └── processed/                  # Processed parquet (gitignored)
├── models/                         # Saved model artifacts (gitignored)
├── notebooks/
│   └── eda.ipynb                   # Exploratory data analysis (11 saved figures)
├── reports/
│   └── figures/                    # EDA plots (PNG)
├── src/
│   ├── __init__.py
│   ├── data_processing.py          # Feature engineering pipeline + RFM + is_high_risk
│   ├── train.py                    # Model training + MLflow tracking
│   ├── predict.py                  # Inference utilities
│   └── api/
│       ├── main.py                 # FastAPI application
│       └── pydantic_models.py      # Request/response schemas
├── tests/
│   └── test_data_processing.py     # 41 unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt                # Full dependencies
├── requirements-api.txt            # API-only dependencies (used in Docker)
└── README.md
```

---

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?

The Basel II Capital Accord requires banks to hold capital reserves proportional to the credit risk they carry. To calculate that risk accurately, banks must demonstrate to regulators that their models are **transparent, auditable, and statistically sound**. This has direct implications for modeling choices:

- **Interpretability is a regulatory requirement, not just a nice-to-have.** Regulators and internal risk teams must be able to understand *why* a model assigns a particular risk score to a borrower. Black-box models that cannot explain their predictions are difficult to defend in a supervisory review.
- **Documentation must trace every modeling decision.** Basel II's Pillar 1 (minimum capital requirements) and Pillar 2 (supervisory review) together demand that the bank document its methodology, validate its assumptions, and demonstrate that the model performs as intended on out-of-sample data.
- **Model risk must be managed.** A poorly documented model that is later found to be flawed can trigger capital add-ons or regulatory sanctions. Full experiment tracking (e.g., via MLflow) and version-controlled code address this by creating a reproducible audit trail.
- **Fair lending considerations.** Models must not produce discriminatory outcomes. Interpretable models (e.g., Logistic Regression with Weight of Evidence) make it easier to audit input features for proxy discrimination.

---

### 2. Without a direct "default" label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?

The raw dataset contains transaction records but **no ground-truth label indicating whether a customer defaulted on a loan**. A proxy variable bridges this gap by using RFM (Recency, Frequency, Monetary) patterns to infer creditworthiness.

**Why a proxy is necessary:**
- Supervised machine learning requires a target label. Without one, no classification model can be trained.
- RFM metrics are well-established indicators of customer engagement and financial health.
- Clustering disengaged customers as "high risk" provides a reasonable approximation of default propensity.

**Business risks introduced by proxy-based prediction:**

| Risk | Description |
|---|---|
| **Label noise** | Some customers labeled "high risk" may be perfectly creditworthy |
| **Concept drift** | The RFM-to-default relationship may change over time |
| **Regulatory exposure** | Basel II expects back-testing against actual default outcomes |
| **Feedback loops** | Denying credit reinforces the high-risk label |
| **Selection bias** | eCommerce customers may not represent the general credit-seeking population |

The proxy variable is a **modeling assumption**, not ground truth, and will be replaced with actual repayment data as it accumulates.

---

### 3. What are the key trade-offs between a simple, interpretable model and a high-performance model in a regulated financial context?

| Dimension | Logistic Regression + WoE | XGBoost |
|---|---|---|
| **Interpretability** | High — coefficients map to log-odds | Low — requires SHAP for explainability |
| **Regulatory compliance** | Easier to defend under Basel II Pillar 2 | Needs additional explainability tooling |
| **Predictive performance** | Moderate (ROC-AUC: 0.933) | High (ROC-AUC: 0.999) |
| **Overfitting risk** | Low | Higher — requires regularization |
| **Scorecard conversion** | Straightforward | Complex |
| **Maintenance** | Easier | More sensitive to data drift |

Both models were trained and tracked in MLflow. XGBoost was selected as the production model based on ROC-AUC. Logistic Regression serves as the interpretable challenger.

---

## Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.828 | 0.390 | 0.891 | 0.543 | 0.933 |
| **XGBoost** | **0.995** | **0.965** | **0.991** | **0.978** | **0.999** |

**Top features (XGBoost):** `txn_month` (44.5%), `transaction_count` (18.2%), `total_value` (6.6%)

---

## EDA Findings

1. **Skewed Transaction Amounts** — Both `Amount` and `Value` are heavily right-skewed. Log1p transformation applied during feature engineering.
2. **Fraud Label Imbalance** — Fraud rate ~0.2%. `FraudResult` cannot proxy credit risk. RFM-based proxy target engineered separately.
3. **Uneven Customer Engagement** — Most customers transact rarely; a small group dominates volume — core signal for RFM clustering.
4. **Temporal Patterns** — Clear intraday and intraweek patterns. `txn_hour` and `txn_day_of_week` extracted as features.
5. **Predictive Categorical Features** — Fraud rates and transaction values vary across product categories and channels.

Full analysis: [`notebooks/eda.ipynb`](notebooks/eda.ipynb)

---

## Setup

```bash
# Clone the repository
git clone https://github.com/RigbeWeleslasie/credit-scoring-model.git
cd credit-scoring-model

# Install dependencies
pip install -r requirements.txt

# Place raw data
cp /path/to/data.csv data/raw/data.csv

# Run feature engineering + RFM pipeline
python3 -m src.data_processing \
  --input data/raw/data.csv \
  --output data/processed/data_processed.parquet

# Train models
python3 -m src.train \
  --data data/processed/data_processed.parquet \
  --experiment credit-risk \
  --mlflow-uri sqlite:///mlflow.db

# Save best model artifact
mkdir -p models
python3 -c "
import mlflow, joblib
mlflow.set_tracking_uri('sqlite:///mlflow.db')
model = mlflow.sklearn.load_model('runs:/<run_id>/model')
joblib.dump(model, 'models/xgboost_model.pkl')
"
```

---

## Running the API

```bash
# With Docker (recommended)
docker build -t credit-risk-api .
docker run -p 8000:8000 \
  -e MODEL_URI="/app/models/xgboost_model.pkl" \
  -v $(pwd)/models:/app/models \
  credit-risk-api

# Without Docker
uvicorn src.api.main:app --reload
```

### Sample Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CustomerId_4406",
    "Amount": 1000.0,
    "Value": 1000.0,
    "PricingStrategy": 2,
    "FraudResult": 0,
    "txn_hour": 14,
    "txn_day": 15,
    "txn_month": 11,
    "txn_year": 2018,
    "txn_day_of_week": 2,
    "total_transaction_amount": 10.5,
    "avg_transaction_amount": 8.2,
    "transaction_count": 15,
    "std_transaction_amount": 2.1,
    "total_value": 10.5,
    "avg_value": 8.2
  }'
```

### Sample Response

```json
{
  "customer_id": "CustomerId_4406",
  "risk_probability": 0.868,
  "risk_label": "high_risk"
}
```

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service and model status |
| `/predict` | POST | Credit risk score for a customer |
| `/docs` | GET | Interactive Swagger UI |

---

## Running Tests

```bash
pytest tests/ -v
# 41 tests passing
```

## MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

---

## CI/CD

Every push triggers:
1. **flake8** — code style linting
2. **pytest** — 41 unit tests
3. **Notebook validation** — structure and cell checks
4. **Required files check** — ensures all project files exist
5. **Docker build** — validates the image builds successfully

---

## Team

- Kerod
- Mahbubah
- Feven

## Key Dates

- **Interim Submission:** Sunday, 31 May 2026, 8:00 PM UTC
- **Final Submission:** Wednesday, 03 Jun 2026, 8:00 PM UTC
