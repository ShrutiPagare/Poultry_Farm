"""Central configuration for the Poultry AI System."""
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "outputs" / "models"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset ──────────────────────────────────────────────
RAW_CSV = DATA_DIR / "poultry_contracts.csv"

# ── Feature groups ───────────────────────────────────────
NUMERIC_FEATURES = [
    "chick_count", "feed_consumed_kg", "mortality_count",
    "contract_duration", "price_per_bird_inr",
    "mortality_rate", "feed_efficiency",
    "revenue_per_bird", "profit_per_bird", "duration_efficiency",
]
CATEGORICAL_FEATURES = ["farm_size", "contract_type", "sale_type", "state"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ── Targets ───────────────────────────────────────────────
TARGET_PROFIT     = "estimated_profit"
TARGET_RISK       = "risk_label"
TARGET_SCORE      = "farm_score"
TARGET_APPROVAL   = "approval"

# ── Train / Test split ────────────────────────────────────
TEST_SIZE  = 0.20
RANDOM_STATE = 42

# ── LSTM ──────────────────────────────────────────────────
SEQUENCE_LENGTH = 6   # months of lookback
LSTM_EPOCHS     = 60
LSTM_BATCH      = 32

# ── ANN Scoring ───────────────────────────────────────────
ANN_EPOCHS = 80
ANN_BATCH  = 32
