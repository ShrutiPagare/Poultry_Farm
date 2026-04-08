"""
Project 3 — Profit Forecasting (LSTM)
Predicts next N months of total profit from historical monthly data.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from configs.config import MODEL_DIR, REPORT_DIR, SEQUENCE_LENGTH, LSTM_EPOCHS, LSTM_BATCH, RANDOM_STATE
from src.preprocessing.data_loader import load_raw, clean, get_time_series_df

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    _TF = True
except ImportError:
    _TF = False
    print("⚠️  TensorFlow not installed. LSTM training skipped.")


def _build_sequences(series: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i: i + seq_len])
        y.append(series[i + seq_len])
    return np.array(X), np.array(y)


def train(forecast_months: int = 6):
    if not _TF:
        return None, None

    df = clean(load_raw())
    ts = get_time_series_df(df)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(ts[["total_profit"]]).flatten()
    joblib.dump(scaler, MODEL_DIR / "lstm_scaler.pkl")

    X, y = _build_sequences(scaled, SEQUENCE_LENGTH)
    X    = X.reshape(X.shape[0], X.shape[1], 1)

    split  = int(len(X) * 0.85)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    tf.random.set_seed(RANDOM_STATE)
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, 1)),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    model.summary()

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5, verbose=0),
    ]
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_te, y_te),
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(MODEL_DIR / "lstm_forecast_model.keras")

    # Forecast future months
    last_seq = scaled[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, 1)
    future_scaled = []
    for _ in range(forecast_months):
        pred       = model.predict(last_seq, verbose=0)[0, 0]
        future_scaled.append(pred)
        last_seq   = np.append(last_seq[:, 1:, :], [[[pred]]], axis=1)

    future_profit = scaler.inverse_transform(
        np.array(future_scaled).reshape(-1, 1)
    ).flatten()

    last_date = ts["month"].max()
    future_months = pd.date_range(last_date, periods=forecast_months + 1, freq="ME")[1:]
    forecast_df = pd.DataFrame({"month": future_months, "forecast_profit": future_profit})
    forecast_df.to_csv(REPORT_DIR / "lstm_forecast.csv", index=False)

    print("✅ LSTM model saved | Forecast:")
    print(forecast_df.to_string(index=False))
    return model, forecast_df


def load_forecast() -> pd.DataFrame:
    path = REPORT_DIR / "lstm_forecast.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["month"])
    return pd.DataFrame(columns=["month", "forecast_profit"])


if __name__ == "__main__":
    train()
