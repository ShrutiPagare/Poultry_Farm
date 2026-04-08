"""
Project 4 — Farm Performance Scoring (ANN)
Output: score 0–100 per farm
"""
import numpy as np
import pandas as pd
import joblib

from configs.config import MODEL_DIR, REPORT_DIR, ANN_EPOCHS, ANN_BATCH, RANDOM_STATE, TARGET_SCORE
from src.preprocessing.data_loader import load_raw, clean

SCORE_FEATURES = [
    "mortality_rate", "feed_efficiency",
    "revenue_per_bird", "profit_per_bird",
    "contract_duration",
]

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    _TF = True
except ImportError:
    _TF = False
    print("⚠️  TensorFlow not installed. ANN training skipped.")


def train():
    if not _TF:
        return None

    df = clean(load_raw())
    X  = df[SCORE_FEATURES].values
    y  = df[TARGET_SCORE].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    joblib.dump(scaler, MODEL_DIR / "ann_score_scaler.pkl")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y, test_size=0.2, random_state=RANDOM_STATE
    )

    tf.random.set_seed(RANDOM_STATE)
    model = Sequential([
        Input(shape=(len(SCORE_FEATURES),)),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),  # output in [0,1]; scale to 0-100
    ])

    def scaled_mse(y_true, y_pred):
        return tf.keras.losses.mse(y_true / 100.0, y_pred)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=scaled_mse)
    model.fit(
        X_tr, y_tr / 100.0,
        validation_split=0.15,
        epochs=ANN_EPOCHS,
        batch_size=ANN_BATCH,
        callbacks=[EarlyStopping(patience=12, restore_best_weights=True)],
        verbose=1,
    )

    y_pred = model.predict(X_te, verbose=0).flatten() * 100
    mae    = mean_absolute_error(y_te, y_pred)
    r2     = r2_score(y_te, y_pred)
    print(f"✅ ANN Score Model — MAE={mae:.2f}  R²={r2:.4f}")

    model.save(MODEL_DIR / "ann_score_model.keras")
    return model


def predict_score(features: dict) -> float:
    """features = dict with SCORE_FEATURES keys"""
    from tensorflow.keras.models import load_model
    model  = load_model(MODEL_DIR / "ann_score_model.keras", compile=False)
    scaler = joblib.load(MODEL_DIR / "ann_score_scaler.pkl")
    X = np.array([[features[f] for f in SCORE_FEATURES]])
    X_sc = scaler.transform(X)
    score = float(model.predict(X_sc, verbose=0)[0, 0]) * 100
    return round(np.clip(score, 0, 100), 1)


if __name__ == "__main__":
    train()
