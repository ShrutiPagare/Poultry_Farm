"""
Project 1 — Profit Prediction (Regression)
Models: LinearRegression | RandomForest | XGBoost
"""
import joblib, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    _XGB = True
except ImportError:
    _XGB = False

from configs.config import MODEL_DIR, REPORT_DIR, TEST_SIZE, RANDOM_STATE, TARGET_PROFIT
from src.preprocessing.data_loader import load_raw, clean, get_X_y


def _metrics(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  [{name}] RMSE={rmse:,.0f}  MAE={mae:,.0f}  R²={r2:.4f}")
    return {"model": name, "RMSE": round(rmse, 2), "MAE": round(mae, 2), "R2": round(r2, 4)}


def train():
    df = clean(load_raw())
    X, y = get_X_y(df, TARGET_PROFIT)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    candidates = {
        "LinearRegression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "RandomForest"    : RandomForestRegressor(n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=RANDOM_STATE),
    }
    if _XGB:
        candidates["XGBoost"] = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                              subsample=0.8, colsample_bytree=0.8,
                                              random_state=RANDOM_STATE, verbosity=0)

    reports, best_r2, best_name, best_model = [], -9e9, None, None
    for name, model in candidates.items():
        model.fit(X_tr, y_tr)
        row = _metrics(y_te, model.predict(X_te), name)
        reports.append(row)
        if row["R2"] > best_r2:
            best_r2, best_name, best_model = row["R2"], name, model

    joblib.dump(best_model, MODEL_DIR / "profit_predictor.pkl")
    report_df = pd.DataFrame(reports)
    report_df.to_csv(REPORT_DIR / "profit_prediction_report.csv", index=False)
    print(f"\n✅ Best model: {best_name}  (R²={best_r2:.4f}) → saved")
    return best_model, report_df


def predict(features: dict):
    model = joblib.load(MODEL_DIR / "profit_predictor.pkl")
    X = pd.DataFrame([features])
    return float(model.predict(X)[0])


if __name__ == "__main__":
    train()
