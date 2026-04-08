"""
Project 5 — Contract Approval Recommendation System (DNN)
Output: Approve / Revise / Reject  (Softmax)
"""
import numpy as np
import pandas as pd
import joblib

from configs.config import MODEL_DIR, REPORT_DIR, RANDOM_STATE, TARGET_APPROVAL, ALL_FEATURES
from src.preprocessing.data_loader import load_raw, clean, encode_categoricals

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    _TF = True
except ImportError:
    _TF = False


APPROVAL_CLASSES = ["Approve", "Reject", "Revise"]


def train():
    if not _TF:
        print("⚠️  TensorFlow not installed."); return None

    df_raw = clean(load_raw())
    df, _  = encode_categoricals(df_raw)
    X      = df[ALL_FEATURES].values

    le = LabelEncoder()
    y  = le.fit_transform(df_raw[TARGET_APPROVAL])
    joblib.dump(le, MODEL_DIR / "approval_label_encoder.pkl")

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    joblib.dump(scaler, MODEL_DIR / "approval_scaler.pkl")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    y_tr_cat = tf.keras.utils.to_categorical(y_tr, num_classes=3)
    y_te_cat = tf.keras.utils.to_categorical(y_te, num_classes=3)

    tf.random.set_seed(RANDOM_STATE)
    model = Sequential([
        Input(shape=(X_sc.shape[1],)),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.35),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.25),
        Dense(64,  activation="relu"),
        Dropout(0.2),
        Dense(32,  activation="relu"),
        Dense(3,   activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        X_tr, y_tr_cat,
        validation_data=(X_te, y_te_cat),
        epochs=80,
        batch_size=32,
        callbacks=[
            EarlyStopping(patience=12, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=6, verbose=0),
        ],
        verbose=1,
    )

    preds = model.predict(X_te, verbose=0).argmax(axis=1)
    acc   = accuracy_score(y_te, preds)
    print(f"\n✅ Approval DNN — Accuracy={acc:.4f}")
    print(classification_report(y_te, preds, target_names=le.classes_))

    model.save(MODEL_DIR / "approval_dnn_model.keras")
    return model


def predict_approval(features: dict) -> dict:
    """Returns {'decision': 'Approve', 'confidence': 0.92, 'probabilities': {...}}"""
    from tensorflow.keras.models import load_model
    model  = load_model(MODEL_DIR / "approval_dnn_model.keras", compile=False)
    scaler = joblib.load(MODEL_DIR / "approval_scaler.pkl")
    le     = joblib.load(MODEL_DIR / "approval_label_encoder.pkl")

    X    = np.array([[features[f] for f in ALL_FEATURES]])
    proba = model.predict(scaler.transform(X), verbose=0)[0]
    idx  = int(np.argmax(proba))
    return {
        "decision"    : le.inverse_transform([idx])[0],
        "confidence"  : round(float(proba[idx]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(le.classes_, proba)},
    }


if __name__ == "__main__":
    train()
