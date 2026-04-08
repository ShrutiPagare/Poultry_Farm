"""
Project 2 — Risk Classification
Labels: Low / Medium / High
Models: LogisticRegression | DecisionTree | RandomForest | XGBoost
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    _XGB = True
except ImportError:
    _XGB = False

from configs.config import MODEL_DIR, REPORT_DIR, TEST_SIZE, RANDOM_STATE, TARGET_RISK
from src.preprocessing.data_loader import load_raw, clean, get_X_y


def train():
    df = clean(load_raw())
    X, y_raw = get_X_y(df, TARGET_RISK)

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    joblib.dump(le, MODEL_DIR / "risk_label_encoder.pkl")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                               stratify=y, random_state=RANDOM_STATE)

    candidates = {
        "LogisticRegression": Pipeline([("sc", StandardScaler()),
                                        ("clf", LogisticRegression(max_iter=500, random_state=RANDOM_STATE))]),
        "DecisionTree"      : DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        "RandomForest"      : RandomForestClassifier(n_estimators=200, max_depth=12,
                                                     class_weight="balanced",
                                                     random_state=RANDOM_STATE, n_jobs=-1),
    }
    if _XGB:
        candidates["XGBoost"] = XGBClassifier(n_estimators=300, learning_rate=0.05,
                                               max_depth=6, use_label_encoder=False,
                                               eval_metric="mlogloss", verbosity=0,
                                               random_state=RANDOM_STATE)

    reports, best_acc, best_name, best_model = [], 0, None, None
    for name, model in candidates.items():
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        acc   = accuracy_score(y_te, preds)
        print(f"\n[{name}] Accuracy = {acc:.4f}")
        print(classification_report(y_te, preds, target_names=le.classes_))
        reports.append({"model": name, "accuracy": round(acc, 4)})
        if acc > best_acc:
            best_acc, best_name, best_model = acc, name, model

    joblib.dump(best_model, MODEL_DIR / "risk_classifier.pkl")
    pd.DataFrame(reports).to_csv(REPORT_DIR / "risk_classification_report.csv", index=False)
    print(f"\n✅ Best model: {best_name}  (Acc={best_acc:.4f}) → saved")
    return best_model


def predict(features: dict) -> str:
    model = joblib.load(MODEL_DIR / "risk_classifier.pkl")
    le    = joblib.load(MODEL_DIR / "risk_label_encoder.pkl")
    X     = pd.DataFrame([features])
    return le.inverse_transform(model.predict(X))[0]


if __name__ == "__main__":
    train()
