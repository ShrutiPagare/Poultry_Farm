"""
Generate realistic synthetic poultry farm dataset.
Run once: python data/generate_dataset.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
N = 2000

states = [
    "Telangana","Bihar","Himachal Pradesh","Sikkim","Uttarakhand",
    "Uttar Pradesh","Odisha","Punjab","Gujarat","Arunachal Pradesh",
    "Maharashtra","Karnataka","Andhra Pradesh","Tamil Nadu","West Bengal"
]
contract_types   = ["Broiler", "Layer", "Breeder"]
sale_types       = ["Wholesale", "Retail", "Export"]
farm_sizes       = ["Small", "Medium", "Large"]

farm_size_map  = {"Small": 1, "Medium": 2, "Large": 3}
size           = np.random.choice(farm_sizes, N, p=[0.4, 0.4, 0.2])
size_factor    = np.array([farm_size_map[s] for s in size])

chick_count        = (size_factor * np.random.randint(3000, 7000, N)).clip(3000, 40000)
feed_per_bird      = np.random.uniform(2.8, 4.5, N)          # kg/bird
feed_consumed      = (chick_count * feed_per_bird).round(1)
mortality_rate_true= np.random.beta(2, 18, N)                 # ~10% avg
mortality_count    = (chick_count * mortality_rate_true).astype(int)
contract_duration  = np.random.randint(30, 180, N)            # days
price_per_bird     = np.random.uniform(180, 260, N)           # INR
revenue_per_bird   = price_per_bird * (1 - mortality_rate_true * 0.5)

feed_cost          = feed_consumed * 22                       # ~22 INR/kg
chick_cost         = chick_count * 45
labor_cost         = chick_count * 8 * (contract_duration / 45)
overhead           = chick_count * 5
total_cost         = feed_cost + chick_cost + labor_cost + overhead
total_revenue      = revenue_per_bird * chick_count
estimated_profit   = total_revenue - total_cost

# Introduce realistic noise
estimated_profit  += np.random.normal(0, total_revenue * 0.05)

# Risk label
def assign_risk(row):
    profit_margin = row["estimated_profit"] / max(row["total_revenue"], 1)
    mort = row["mortality_rate"]
    if profit_margin < 0.03 or mort > 0.18:
        return "High"
    elif profit_margin < 0.10 or mort > 0.12:
        return "Medium"
    return "Low"

# Farm score (0-100)
def farm_score(row):
    feed_eff  = max(0, 1 - (row["feed_efficiency"] - 2.8) / (4.5 - 2.8))
    mort_eff  = max(0, 1 - row["mortality_rate"] / 0.25)
    prof_eff  = min(1, max(0, (row["profit_per_bird"] + 20) / 120))
    score = (feed_eff * 35 + mort_eff * 35 + prof_eff * 30)
    return round(score, 1)

df = pd.DataFrame({
    "farm_id"           : [f"F{str(i).zfill(4)}" for i in range(1, N+1)],
    "state"             : np.random.choice(states, N),
    "farm_size"         : size,
    "contract_type"     : np.random.choice(contract_types, N, p=[0.6,0.3,0.1]),
    "sale_type"         : np.random.choice(sale_types, N, p=[0.5,0.35,0.15]),
    "chick_count"       : chick_count.astype(int),
    "feed_consumed_kg"  : feed_consumed.round(1),
    "mortality_count"   : mortality_count,
    "contract_duration" : contract_duration,
    "price_per_bird_inr": price_per_bird.round(2),
    "total_revenue"     : total_revenue.round(2),
    "total_cost"        : total_cost.round(2),
    "estimated_profit"  : estimated_profit.round(2),
})

df["mortality_rate"]   = (df["mortality_count"] / df["chick_count"]).round(4)
df["feed_efficiency"]  = (df["feed_consumed_kg"] / df["chick_count"]).round(4)
df["revenue_per_bird"] = (df["total_revenue"] / df["chick_count"]).round(2)
df["profit_per_bird"]  = (df["estimated_profit"] / df["chick_count"]).round(2)
df["duration_efficiency"] = (df["estimated_profit"] / df["contract_duration"]).round(2)
df["risk_label"]       = df.apply(assign_risk, axis=1)
df["farm_score"]       = df.apply(farm_score, axis=1)
df["approval"]         = df["risk_label"].map({"Low":"Approve","Medium":"Revise","High":"Reject"})

# Encode categoricals for modelling
for col in ["farm_size","contract_type","sale_type","state"]:
    df[col] = df[col].astype("category")

out = Path(__file__).parent / "poultry_contracts.csv"
df.to_csv(out, index=False)
print(f"✅ Dataset saved: {out}  |  Shape: {df.shape}")
print(df.describe().T[["mean","min","max"]].round(2))
