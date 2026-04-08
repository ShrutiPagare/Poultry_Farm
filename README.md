# 🐔 Poultry Farm AI Intelligence System

> **End-to-end AI platform** for poultry contract risk analysis, profit prediction, farm scoring, and automated decision-making.

---

## 📁 Project Structure

```
poultry_ai_system/
│
├── 📂 data/
│   ├── generate_dataset.py          # Synthetic data generator
│   └── poultry_contracts.csv        # Generated dataset (2000 rows)
│
├── 📂 notebooks/
│   └── 01_EDA.ipynb                 # Exploratory Data Analysis
│
├── 📂 src/
│   ├── 📂 preprocessing/
│   │   └── data_loader.py           # Loading, cleaning, encoding
│   ├── 📂 models/
│   │   ├── profit_prediction.py     # Project 1 — XGBoost regressor
│   │   ├── risk_classification.py   # Project 2 — RF classifier
│   │   ├── lstm_forecasting.py      # Project 3 — LSTM time series
│   │   ├── ann_farm_scoring.py      # Project 4 — ANN scorer
│   │   └── approval_recommendation.py  # Project 5 — DNN decision
│   └── 📂 utils/
│       └── evaluation.py            # Metrics & plot helpers
│
├── 📂 configs/
│   └── config.py                    # Central config (paths, params)
│
├── 📂 outputs/
│   ├── models/                      # Saved .pkl / .keras models
│   └── reports/                     # CSV evaluation reports
│
├── 📂 tests/
│   └── test_models.py               # pytest unit tests
│
├── app.py                           # 🚀 Streamlit Dashboard
├── train_all.py                     # Master training script
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate dataset
```bash
python data/generate_dataset.py
```

### 3. Train all models
```bash
python train_all.py
```

### 4. Launch dashboard
```bash
streamlit run app.py
```

---

## 🧠 AI Models

| # | Project | Algorithm | Target | Metrics |
|---|---------|-----------|--------|---------|
| 1 | Profit Prediction | XGBoost Regressor | `estimated_profit` | RMSE, MAE, R² |
| 2 | Risk Classification | Random Forest | `risk_label` (Low/Med/High) | Accuracy, F1 |
| 3 | Profit Forecasting | LSTM | Monthly profit series | RMSE, MAPE |
| 4 | Farm Scoring | ANN (MLP) | `farm_score` 0–100 | MAE, R² |
| 5 | Approval Decision | Deep Neural Network | Approve/Revise/Reject | Accuracy, Confidence |

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Dashboard | KPI summary, risk & approval distribution, state-level charts |
| 📊 EDA | Interactive distributions, correlations, segment analysis |
| 💰 Profit Predictor | Real-time profit estimation with cost breakdown |
| ⚠️ Risk Classifier | Per-contract risk scoring + portfolio heatmap |
| 🏆 Farm Scorer | 0-100 performance scoring + leaderboard |
| 🤖 Decision Engine | Approve/Revise/Reject recommendations |
| 📈 Profit Forecast | LSTM 6-month forecast with confidence intervals |

---

## ⚙️ Feature Engineering

| Feature | Formula |
|---------|---------|
| `mortality_rate` | `mortality_count / chick_count` |
| `feed_efficiency` | `feed_consumed_kg / chick_count` |
| `revenue_per_bird` | `total_revenue / chick_count` |
| `profit_per_bird` | `estimated_profit / chick_count` |
| `duration_efficiency` | `estimated_profit / contract_duration` |

---

## 🧪 Run Tests
```bash
pytest tests/ -v
```

---

## 🛠 Tech Stack
- **ML/DL:** scikit-learn, XGBoost, TensorFlow/Keras
- **Data:** pandas, NumPy
- **Viz:** Plotly, Seaborn, Matplotlib
- **UI:** Streamlit
- **Tests:** pytest
