# Credit Scoring Model

An end-to-end credit risk probability model built for Bati Bank's buy-now-pay-later partnership with an eCommerce platform. This project transforms raw transaction data into a deployed model service that scores new applicants in real time.

---

## Project Structure

```
credit-scoring-model/
├── .github/workflows/ci.yml       # CI/CD pipeline
├── data/
│   ├── raw/                        # Raw data (gitignored)
│   └── processed/                  # Processed data (gitignored)
├── notebooks/
│   └── eda.ipynb                   # Exploratory data analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py          # Feature engineering pipeline
│   ├── train.py                    # Model training & MLflow tracking
│   ├── predict.py                  # Inference utilities
│   └── api/
│       ├── main.py                 # FastAPI application
│       └── pydantic_models.py      # Request/response schemas
├── tests/
│   └── test_data_processing.py     # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
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

In summary, Basel II pushes practitioners toward models where every parameter, feature, and transformation can be explained to a non-technical auditor — making interpretability a first-class engineering constraint.

---

### 2. Without a direct "default" label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?

The raw dataset contains transaction records but **no ground-truth label indicating whether a customer defaulted on a loan**. This is common when a lender is new to a market or when historical loan repayment data has not been collected. A proxy variable bridges this gap by using observable behavioral signals — in this case, RFM (Recency, Frequency, Monetary) patterns — to infer creditworthiness.

**Why a proxy is necessary:**
- Supervised machine learning requires a target label. Without one, no classification model can be trained.
- RFM metrics are well-established in both marketing and credit risk literature as indicators of customer engagement and financial health. A customer who transacts frequently, recently, and with high monetary value is behaviorally similar to a low-risk borrower.
- Clustering disengaged customers (low recency, low frequency, low monetary) as "high risk" provides a reasonable, defensible approximation of default propensity.

**Business risks introduced by proxy-based prediction:**

| Risk | Description |
|---|---|
| **Label noise** | The proxy is not ground truth. Some customers labeled "high risk" may be perfectly creditworthy, and vice versa. |
| **Concept drift** | The relationship between RFM behavior and actual default may change over time, making the proxy unstable. |
| **Regulatory exposure** | Basel II expects models to be validated against actual default outcomes. A proxy-based model may not satisfy back-testing requirements long-term. |
| **Feedback loops** | Denying credit to customers labeled high-risk prevents them from demonstrating creditworthiness, reinforcing the label. |
| **Selection bias** | The eCommerce platform's customer base may not represent the general credit-seeking population. |

The proxy variable should be treated as a **modeling assumption**, not ground truth. As actual loan repayment data accumulates, the proxy should be replaced with real default labels and the model retrained.

---

### 3. What are the key trade-offs between a simple, interpretable model and a high-performance model in a regulated financial context?

| Dimension | Logistic Regression + WoE | Gradient Boosting (XGBoost / LightGBM) |
|---|---|---|
| **Interpretability** | High — coefficients map directly to log-odds; WoE bins are explainable to regulators | Low — hundreds of trees; no direct feature-level explanation without SHAP |
| **Regulatory compliance** | Easier to document and defend under Basel II Pillar 2 | Requires additional explainability tooling (SHAP, LIME) to satisfy auditors |
| **Predictive performance** | Moderate — assumes linear log-odds relationship | High — captures non-linear interactions and complex patterns |
| **Feature engineering burden** | High — requires careful WoE binning, IV selection, monotonicity checks | Lower — tree models handle raw features, missing values, and skew natively |
| **Overfitting risk** | Low — simpler model, fewer parameters | Higher — requires careful regularization and cross-validation |
| **Scorecard conversion** | Straightforward — logistic regression maps cleanly to a points-based scorecard | Complex — non-linear outputs are harder to convert to a scorecard format |
| **Maintenance** | Easier — fewer hyperparameters, stable behavior | Harder — more sensitive to data drift, requires more monitoring |

**Recommendation for this project:** Train both a Logistic Regression (with WoE) and a Gradient Boosting model. Use the Logistic Regression as the **primary production model** for Basel II compliance and interpretability, and the Gradient Boosting model as a **challenger model** to benchmark performance. If the performance gap is large, document the trade-off and consider using SHAP values to partially satisfy explainability requirements for the challenger.

---

## EDA Findings

The exploratory analysis of the Xente transaction dataset revealed the following key insights:

1. **Skewed Transaction Amounts** — Both `Amount` and `Value` are heavily right-skewed with extreme outliers. Log transformation will be applied during feature engineering.

2. **Fraud Label Imbalance** — The fraud rate is ~0.2%, making `FraudResult` unsuitable as a credit risk proxy. A separate target variable will be engineered using RFM segmentation.

3. **Uneven Customer Engagement** — Transaction frequency per customer is highly skewed. A small group of highly active customers dominates, while most customers have very few transactions — the core signal for RFM-based risk labeling.

4. **Temporal Patterns** — Clear intraday and intraweek transaction patterns exist. Hour of day and day of week will be extracted as features.

5. **Predictive Categorical Features** — Fraud rates and transaction values vary significantly across product categories and channels, making them valuable features for the model.

Full analysis: [`notebooks/eda.ipynb`](notebooks/eda.ipynb)

## Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/credit-scoring-model.git
cd credit-scoring-model

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Place raw data
cp /path/to/data.csv data/raw/data.csv
```

## Running the API

```bash
# With Docker
docker-compose up --build

# Without Docker
uvicorn src.api.main:app --reload
```

## Running Tests

```bash
pytest tests/
```

## MLflow UI

```bash
mlflow ui
# Open http://localhost:5000
```

---

## Team

- Kerod
- Mahbubah
- Feven

## Key Dates

- **Interim Submission:** Sunday, 31 May 2026, 8:00 PM UTC
- **Final Submission:** Wednesday, 03 Jun 2026, 8:00 PM UTC