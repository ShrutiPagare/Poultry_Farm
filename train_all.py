"""
Master training pipeline — runs all 5 models sequentially.
Usage:  python train_all.py
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data.generate_dataset import *   # ensure CSV exists
from src.models.profit_prediction   import train as train_profit
from src.models.risk_classification import train as train_risk
from src.models.lstm_forecasting    import train as train_lstm
from src.models.ann_farm_scoring    import train as train_ann
from src.models.approval_recommendation import train as train_approval


def main():
    steps = [
        ("1/5  Profit Prediction (ML)",          train_profit),
        ("2/5  Risk Classification (ML)",         train_risk),
        ("3/5  Profit Forecasting (LSTM)",        lambda: train_lstm(forecast_months=6)),
        ("4/5  Farm Scoring (ANN)",               train_ann),
        ("5/5  Approval Recommendation (DNN)",    train_approval),
    ]
    total_start = time.time()
    for label, fn in steps:
        print(f"\n{'='*60}")
        print(f"  ▶  {label}")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            fn()
        except Exception as e:
            print(f"  ⚠️  {label} FAILED: {e}")
        print(f"  ⏱  Done in {time.time()-t0:.1f}s")

    print(f"\n🎉 All models trained in {time.time()-total_start:.1f}s")
    print("   ➜  Saved to: outputs/models/")


if __name__ == "__main__":
    main()
