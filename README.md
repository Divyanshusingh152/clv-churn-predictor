# 📡 Customer Lifetime Value & Churn Predictor

A full end-to-end data analytics and machine learning project that predicts customer churn and calculates Customer Lifetime Value (CLV) for a Telco company — built with Python, SQL, Scikit-learn, Plotly and Streamlit.

🔴 **Live App:** [https://clv-churn-predictor-anyur9slnappek5vy5rkgnm.streamlit.app](https://clv-churn-predictor-anyur9slnappek5vy5rkgnm.streamlit.app)

---

## 📌 Business Problem

Imagine you work at Airtel or Jio. Every month hundreds of customers cancel their plan and move to a competitor. Each lost customer = ₹500–2000/month lost forever.

This project answers two critical business questions:
- **WHO is about to leave** before they actually leave — so the company can offer a retention discount in time
- **WHICH customers are most valuable** — so the company knows who to fight for

---

## 🎯 Key Findings

| Finding | Detail |
|---|---|
| Overall Churn Rate | **26.54%** — 1 in 4 customers leaving |
| Highest Risk Group | Month-to-month contracts churn at **42.71%** |
| Safest Group | Two year contracts churn at only **2.83%** |
| Service Risk | Fiber Optic customers churn at **41.89%** — drives 69% of all churn |
| Payment Risk | Electronic check customers churn at **45.29%** — 3x higher than auto-pay |
| Tenure Risk | **47.44%** of new customers (0–12 months) churn — 5x higher than loyal customers |
| High Value Churn | Churned customers pay **₹13 more/month** on average than retained ones |
| CLV Gap | Churned Two Year customers had **49% higher CLV** (₹5,441 vs ₹3,656) |

---

## 🔍 Typical Churned Customer Profile

```
Tenure          → Less than 18 months
Monthly Charges → ₹74.44/month (above average)
Contract        → Month-to-month (88.55% of churners)
Internet        → Fiber Optic (69.40% of churners)
Payment         → Electronic Check (57.30% of churners)
Senior Citizen  → 25.47% are seniors (over-represented)
```

---

## 🛠️ Tools & Technologies

| Tool | Purpose |
|---|---|
| **Python** | Core programming language |
| **Pandas** | Data loading, cleaning, EDA |
| **PostgreSQL + SQLAlchemy** | Database storage and SQL analysis |
| **Scikit-learn** | Random Forest ML model |
| **Plotly** | Interactive charts |
| **Streamlit** | Live web application |

---

## 📊 Project Phases

### Phase 1 — Pandas EDA
- Loaded and cleaned 7,043 customer records
- Fixed TotalCharges dtype, SeniorCitizen encoding
- Answered 8 business questions using Pandas
- Identified key churn drivers across demographics, services and contract types

### Phase 2 — SQL Analysis (PostgreSQL)
- Loaded cleaned data into PostgreSQL using SQLAlchemy
- Wrote 6 advanced SQL queries including:
  - **CTE** — Average CLV per customer segment
  - **Window Function** — Revenue ranking within contract groups
  - **Cohort Analysis** — Churn rate by tenure group

### Phase 3 — Machine Learning (Scikit-learn)
- Encoded 29 features using One Hot Encoding
- Split data 80/20 train/test
- Trained Random Forest Classifier (100 trees, max depth 10)
- Achieved **81.05% accuracy** on unseen test data
- Top churn predictor: **tenure (24.7%)** followed by MonthlyCharges (13.7%)

### Phase 4 — Plotly Charts
- Chart 1: Churn rate by contract type (interactive bar chart)
- Chart 2: Feature importance (horizontal bar chart)
- Chart 3: CLV distribution churned vs retained (histogram)

### Phase 5 — Streamlit App + Deployment
- Built 4-section live web app
- Section 1: Key business metrics dashboard
- Section 2: Interactive Plotly charts
- Section 3: Feature importance visualization
- Section 4: Live churn predictor — enter customer details → instant prediction
- Deployed FREE on Streamlit Cloud

---

## 🤖 ML Model Performance

```
Algorithm        : Random Forest Classifier
Trees            : 100
Accuracy         : 81.05%
Churners Caught  : 51% (190 of 373 test churners)

Confusion Matrix:
                 Predicted Stay   Predicted Churn
Actually Stay  →     952               84
Actually Churn →     183              190
```

---

## 📁 Project Structure

```
clv-churn-predictor/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│   ├── 01_data_loading_and_exploration.ipynb
│   ├── 02_ml_churn_prediction.ipynb
│   └── 03_plotly_charts.ipynb
│
├── app/
│   └── app.py
│
└── requirements.txt
```

---

## 🚀 How To Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Divyanshusingh152/clv-churn-predictor.git
cd clv-churn-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Streamlit app**
```bash
cd app
python -m streamlit run app.py
```

**4. Open in browser**
```
http://localhost:8501
```

---

## 📋 SQL Queries Included

| Query | Type | Business Question |
|---|---|---|
| Q9 | GROUP BY | Churn rate per contract type |
| Q10 | ORDER BY + LIMIT | Top 10 most valuable customers |
| Q11 | CTE | Average CLV per customer segment |
| Q12 | GROUP BY | Churn rate by payment method |
| Q13 | Window Function | Revenue ranking within contract groups |
| Q14 | CTE + CASE WHEN | Cohort analysis by tenure group |

---

## 💼 Resume Bullet Point

> *"Built end-to-end Customer Lifetime Value & Churn Predictor using Pandas, PostgreSQL, Scikit-learn (Random Forest, 81% accuracy), Plotly and Streamlit — analysed 7,043 Telco customer records, identified 26.54% churn rate with key drivers (contract type, tenure, service type), deployed live at https://clv-churn-predictor-anyur9slnappek5vy5rkgnm.streamlit.app"*

---

## 👨‍💻 Author

**Divyanshu Singh**
- GitHub: [@Divyanshusingh152](https://github.com/Divyanshusingh152)

---

## 📄 Dataset

- **Source:** [Telco Customer Churn — IBM Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Records:** 7,043 customers
- **Features:** 21 columns
- **Target:** Churn (Yes/No)

---

*Project 2 of Data Analyst Portfolio — building on [Project 1: Job Market Intelligence Platform](https://github.com/Divyanshusingh152/job-market-intelligence-platform)*
