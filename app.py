"""
╔══════════════════════════════════════════════════════════════╗
║  🐔  Poultry Farm AI Intelligence — Unified Dashboard v3.0  ║
║  All inputs in sidebar · All outputs in one scrolling view  ║
╚══════════════════════════════════════════════════════════════╝
Run:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="PoultryAI — Farm Intelligence",
    page_icon="🐔",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.write("App is running successfully ✅")


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #0D0A18 !important;
    color: #E2D9F3;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
section.main, .main, div.main {
    background: #0D0A18 !important;
}
[data-testid="stHeader"] { background: #0D0A18 !important; }
[data-testid="stDecoration"] { display: none; }
.stTabs [data-baseweb="tab-list"] { background: transparent !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: #9B7EC8 !important; }
.stTabs [aria-selected="true"] {
    background: rgba(192,132,252,0.1) !important;
    color: #E879F9 !important; border-bottom: 2px solid #C084FC !important;
}
.stDataFrame thead tr th {
    background: #1A0E2E !important; color: #9B7EC8 !important;
    font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.06em;
}
div[data-testid="stExpander"] {
    background: #130A22 !important; border: 1px solid rgba(192,132,252,0.1) !important;
    border-radius: 10px !important;
}
hr { border-color: rgba(192,132,252,0.08) !important; }

/* ─── Main Container ──────────────────────────────────────── */
.main .block-container {
    padding: 1.2rem 2rem 3rem !important;
    max-width: 1600px;
}

/* ─── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0E0818 0%, #130B20 100%) !important;
    border-right: 1px solid rgba(192,132,252,0.12);
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0; }
[data-testid="stSidebar"] * { color: #E2D9F3 !important; }
[data-testid="stSidebarContent"] { padding: 0 !important; }
[data-testid="stSidebar"] .stSlider > div > div > div { background: #C084FC !important; }
[data-testid="stSidebar"] .stSlider > div > div { background: rgba(192,132,252,0.15) !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.07) !important; margin: 10px 0 !important; }

/* ─── Number Inputs ───────────────────────────────────────── */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    background: #1A0E2E !important; border: 1px solid rgba(192,132,252,0.25) !important;
    color: #E2D9F3 !important; border-radius: 8px !important; font-size: 0.95rem !important;
}
div[data-testid="stNumberInput"] button {
    background: #220F3A !important; border: 1px solid rgba(192,132,252,0.2) !important;
    color: #E2D9F3 !important; border-radius: 6px !important;
}
div[data-testid="stNumberInput"] button:hover { background: #2D1550 !important; }

/* ─── Selectbox ───────────────────────────────────────────── */
.stSelectbox [data-baseweb="select"] > div {
    background: #1A0E2E !important; border: 1px solid rgba(192,132,252,0.25) !important;
    color: #E2D9F3 !important; border-radius: 8px !important;
}
[data-baseweb="popover"] { background: #1A0E2E !important; border: 1px solid rgba(192,132,252,0.2) !important; }
[data-baseweb="menu"] { background: #1A0E2E !important; }
[data-baseweb="option"] { background: #1A0E2E !important; color: #E2D9F3 !important; }
[data-baseweb="option"]:hover { background: #2D1550 !important; }

/* ─── Button ──────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #C084FC 0%, #E879F9 50%, #F472B6 100%) !important;
    color: #0D0A18 !important; font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important; font-size: 0.9rem !important;
    border: none !important; border-radius: 10px !important;
    padding: 14px 20px !important; width: 100% !important;
    letter-spacing: 0.04em; transition: all 0.25s ease !important;
    box-shadow: 0 4px 28px rgba(232,121,249,0.40) !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 8px 36px rgba(232,121,249,0.60) !important;
}

/* ─── Metric Cards ────────────────────────────────────────── */
div[data-testid="metric-container"] {
    background: #111827; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 16px 20px;
}

/* ─── KPI Cards ───────────────────────────────────────────── */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 22px; }
.kpi-card {
    background: #111827; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 22px 24px;
    position: relative; overflow: hidden; transition: transform 0.22s, border-color 0.22s, box-shadow 0.22s;
    cursor: default;
}
.kpi-card:hover { transform: translateY(-3px); border-color: rgba(192,132,252,0.45); box-shadow: 0 8px 32px rgba(192,132,252,0.15); }
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--accent, linear-gradient(90deg,#C084FC,#E879F9));
}
.kpi-card::after {
    content: ''; position: absolute; top: -40px; right: -40px;
    width: 120px; height: 120px; border-radius: 50%;
    background: var(--glow, radial-gradient(circle, rgba(192,132,252,0.10) 0%, transparent 70%));
    pointer-events: none;
}
.kpi-icon { font-size: 1.6rem; margin-bottom: 10px; display: block; }
.kpi-label {
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #64748B; margin-bottom: 6px;
}
.kpi-value {
    font-family: 'Syne', sans-serif; font-size: 1.75rem; font-weight: 800;
    color: #F1F5F9; line-height: 1; margin-bottom: 6px;
}
.kpi-sub { font-size: 0.78rem; font-weight: 500; color: #64748B; }
.kpi-badge {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 0.82rem; font-weight: 600; padding: 4px 12px;
    border-radius: 20px; margin-top: 4px;
}

/* ─── Section Header ──────────────────────────────────────── */
.section-title {
    font-family: 'Syne', sans-serif; font-size: 1.25rem; font-weight: 800;
    color: #F1F5F9; letter-spacing: -0.02em; margin: 0 0 14px;
    display: flex; align-items: center; gap: 10px;
}
.section-title .line {
    flex: 1; height: 1px; background: rgba(255,255,255,0.07); margin-left: 8px;
}

/* ─── Chart Card ──────────────────────────────────────────── */
.chart-card {
    background: #130A22; border: 1px solid rgba(192,132,252,0.10);
    border-radius: 16px; padding: 20px 22px; margin-bottom: 16px;
}

/* ─── Sidebar Section Label ───────────────────────────────── */
.sb-section {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #9B7EC8;
    padding: 14px 20px 6px; display: flex; align-items: center; gap: 8px;
}
.sb-section::after {
    content: ''; flex: 1; height: 1px; background: rgba(192,132,252,0.15);
}

/* ─── Classification Badge ────────────────────────────────── */
.class-badge {
    padding: 14px 20px; border-radius: 10px; font-family: 'Syne', sans-serif;
    font-size: 1rem; font-weight: 700; display: flex; align-items: center; gap: 10px;
    margin-bottom: 10px;
}

/* ─── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0D0A18; }
::-webkit-scrollbar-thumb { background: #2D1550; border-radius: 3px; }

/* ─── Plotly toolbar hide ─────────────────────────────────── */
.js-plotly-plot .plotly .modebar { opacity: 0.3; }
.js-plotly-plot .plotly .modebar:hover { opacity: 1; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CHART THEME
# ─────────────────────────────────────────────────────────────
CHART_BG   = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(192,132,252,0.07)"
TEXT_COLOR = "#9B7EC8"
FONT       = "DM Sans"
C_YELLOW   = "#C084FC"   # primary purple
C_GREEN    = "#4ADE80"
C_RED      = "#F87171"
C_ORANGE   = "#E879F9"   # magenta/pink
C_CYAN     = "#818CF8"   # indigo
C_PURPLE   = "#F472B6"   # hot pink
RISK_C     = {"Low": C_GREEN, "Medium": "#E879F9", "High": C_RED}
APPR_C     = {"Approve": C_GREEN, "Revise": "#C084FC", "Reject": C_RED}


def chart_layout(fig, h=320, xlab="", ylab="", show_legend=False):
    fig.update_layout(
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        font=dict(family=FONT, color=TEXT_COLOR, size=12),
        height=h,
        margin=dict(t=20, b=40, l=50, r=20),
        xaxis=dict(
            gridcolor=GRID_COLOR, showline=False, zeroline=False,
            tickfont=dict(color=TEXT_COLOR, size=11), title=dict(text=xlab, font=dict(color=TEXT_COLOR)),
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR, showline=False, zeroline=False,
            tickfont=dict(color=TEXT_COLOR, size=11), title=dict(text=ylab, font=dict(color=TEXT_COLOR)),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8", size=11)) if show_legend else dict(visible=False),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1A0E2E", bordercolor="rgba(192,132,252,0.3)", font=dict(color="#E2D9F3", size=12)),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = Path(__file__).parent / "data" / "poultry_contracts.csv"
    if path.exists():
        df = pd.read_csv(path)
        for c in ["farm_size", "contract_type", "sale_type", "state", "risk_label", "approval"]:
            if c in df.columns:
                df[c] = df[c].astype(str)
        return df
    return None


# ─────────────────────────────────────────────────────────────
# SIDEBAR — ALL INPUTS
# ─────────────────────────────────────────────────────────────
def render_sidebar(df):
    with st.sidebar:
        # Brand Header
        st.markdown("""
        <div style="padding:22px 20px 16px;border-bottom:1px solid rgba(192,132,252,0.1);
                    background:linear-gradient(135deg,rgba(192,132,252,0.06),transparent)">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px">
                <span style="font-size:2rem;filter:drop-shadow(0 0 10px rgba(232,121,249,0.7))">🐔</span>
                <div>
                    <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.2rem;
                         background:linear-gradient(90deg,#E879F9,#C084FC,#818CF8);
                         -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                         line-height:1.1">PoultryAI</div>
                    <div style="font-size:0.62rem;color:#6B4F8A;letter-spacing:0.1em;
                         text-transform:uppercase;margin-top:2px">Farm Intelligence v3.0</div>
                </div>
            </div>
            <div style="margin-top:10px;padding:6px 10px;background:rgba(192,132,252,0.10);
                 border:1px solid rgba(192,132,252,0.25);border-radius:6px;
                 font-size:0.68rem;color:#C084FC;letter-spacing:0.06em">
                ✦ AI Powered Farm Revenue Intelligence
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Location Details ──────────────────────────────────
        st.markdown("<div class='sb-section'>📍 Location Details</div>", unsafe_allow_html=True)
        with st.container():
            states_list = ["All States"] + (sorted(df["state"].unique().tolist()) if df is not None else [])
            sel_state = st.selectbox("Select State", states_list, key="g_state")

        # ── Contract Configuration ─────────────────────────────
        st.markdown("<div class='sb-section'>📋 Contract Configuration</div>", unsafe_allow_html=True)
        sel_contract = st.selectbox("Contract Type", ["Fixed Price Contract", "Cost Plus Contract", "Performance Contract", "Hybrid Contract"], key="contract_type")
        sel_sale     = st.selectbox("Sale Type", ["Wholesale", "Retail", "Export"], key="sale_type")
        sel_farmsize = st.selectbox("Farm Size", ["Small", "Medium", "Large"], key="farm_size", index=1)

        # ── Farm Inputs ────────────────────────────────────────
        st.markdown("<div class='sb-section'>🐔 Farm Inputs</div>", unsafe_allow_html=True)
        total_birds  = st.number_input("Total Bird Count", min_value=500, max_value=50000, value=2000, step=100, key="birds")
        feed_cost    = st.number_input("Feed Cost (₹ per bird)", min_value=50, max_value=500, value=120, step=5, key="feed_cost")
        mortality    = st.slider("Mortality Rate (%)", min_value=0.0, max_value=30.0, value=5.0, step=0.5, key="mortality")
        sell_price   = st.number_input("Market Selling Price (₹)", min_value=100, max_value=500, value=220, step=5, key="sell_price")

        # ── Revenue & Financial Inputs ─────────────────────────
        st.markdown("<div class='sb-section'>💰 Financial Inputs</div>", unsafe_allow_html=True)
        farm_size_n  = st.number_input("Farm Size (Bird Capacity)", min_value=500, max_value=50000, value=5000, step=500, key="farm_cap")
        experience   = st.slider("Experience (Years)", min_value=0, max_value=30, value=5, key="exp")
        total_chicks = st.number_input("Total Chicks", min_value=500, max_value=50000, value=4000, step=500, key="chicks")
        feed_total   = st.number_input("Total Feed Cost (₹)", min_value=10000, max_value=5000000, value=200000, step=10000, key="feed_total")
        total_revenue= st.number_input("Total Revenue (₹)", min_value=10000, max_value=10000000, value=800000, step=10000, key="total_rev")
        contract_dur = st.slider("Contract Duration (Months)", min_value=1, max_value=24, value=12, key="contract_dur")
        exp_roi      = st.slider("Expected ROI (%)", min_value=0.0, max_value=50.0, value=15.0, step=0.5, key="exp_roi")

        st.markdown("<hr style='margin:16px 0'>", unsafe_allow_html=True)

        run_btn = st.button("🚀 RUN SMART PREDICTION", key="run_btn")

        # ── Portfolio Quick Stats ──────────────────────────────
        if df is not None:
            st.markdown("<div class='sb-section'>📊 Live Portfolio</div>", unsafe_allow_html=True)
            total_r  = df["total_revenue"].sum()
            avg_roi  = ((df["estimated_profit"] / df["total_revenue"]) * 100).mean()
            high_r   = (df["risk_label"] == "High").mean() * 100
            approve  = (df["approval"] == "Approve").mean() * 100
            for lbl, val, color in [
                ("Revenue",       f"₹{total_r/1e8:.2f}Cr", C_YELLOW),
                ("Avg ROI",       f"{avg_roi:.1f}%",        C_GREEN),
                ("High Risk",     f"{high_r:.1f}%",         C_RED),
                ("Approval Rate", f"{approve:.1f}%",        C_CYAN),
            ]:
                st.markdown(f"""
                <div style='display:flex;justify-content:space-between;align-items:center;
                     padding:7px 10px;background:rgba(255,255,255,0.03);border-radius:7px;
                     margin-bottom:4px;border-left:3px solid {color}'>
                    <span style='font-size:0.74rem;color:#9B7EC8'>{lbl}</span>
                    <span style='font-size:0.86rem;font-weight:700;font-family:Syne,sans-serif;color:{color}'>{val}</span>
                </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style='padding:16px 12px 8px;text-align:center'>
            <div style='font-size:0.6rem;color:#1E293B;letter-spacing:0.06em'>
                XGBoost · LSTM · ANN · DNN · Random Forest<br>
                <span style='color:#0F172A'>© 2025 PoultryAI Intelligence</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    return {
        "state": sel_state, "contract": sel_contract, "sale": sel_sale,
        "farm_size_cat": sel_farmsize,
        "birds": total_birds, "feed_cost": feed_cost, "mortality": mortality, "sell_price": sell_price,
        "farm_cap": farm_size_n, "experience": experience, "chicks": total_chicks,
        "feed_total": feed_total, "revenue": total_revenue,
        "duration": contract_dur, "roi": exp_roi,
        "run": run_btn,
    }


# ─────────────────────────────────────────────────────────────
# COMPUTATION ENGINE
# ─────────────────────────────────────────────────────────────
def compute_predictions(inp):
    mort_r    = inp["mortality"] / 100
    survivors = inp["birds"] * (1 - mort_r)
    revenue   = inp["sell_price"] * survivors
    feed_c    = inp["feed_cost"] * inp["birds"]
    other_c   = inp["birds"] * 30
    tot_cost  = feed_c + other_c
    profit    = revenue - tot_cost
    margin    = profit / max(revenue, 1)

    # Farm score (ANN-style composite)
    mort_sc = max(0, 1 - mort_r / 0.25)
    feed_sc = min(1, max(0, 1 - (inp["feed_cost"] - 80) / 200))
    rev_sc  = min(1, max(0, (inp["sell_price"] - 150) / 150))
    exp_sc  = min(1, inp["experience"] / 20)
    score   = round(mort_sc * 35 + feed_sc * 25 + rev_sc * 25 + exp_sc * 15, 1)

    # Risk
    if mort_r > 0.18 or margin < 0.03:
        risk = "High"
    elif mort_r > 0.10 or margin < 0.10:
        risk = "Medium"
    else:
        risk = "Low"

    # Decision
    if margin >= 0.10 and mort_r <= 0.12 and score >= 55:
        decision = "Approve"
        conf = min(0.97, 0.75 + margin)
    elif margin < 0.02 or mort_r > 0.22 or score < 30:
        decision = "Reject"
        conf = 0.88
    else:
        decision = "Revise"
        conf = 0.74

    return {
        "profit": profit, "revenue": revenue, "margin": margin,
        "score": score, "risk": risk, "decision": decision, "conf": conf,
        "mort_r": mort_r, "survivors": survivors, "tot_cost": tot_cost,
        "mort_sc": mort_sc, "feed_sc": feed_sc, "rev_sc": rev_sc, "exp_sc": exp_sc,
    }


# ─────────────────────────────────────────────────────────────
# MAIN DASHBOARD SECTIONS
# ─────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div style="background:linear-gradient(135deg,#120920 0%,#1a0e2e 60%,#0f0a1e 100%);
         border:1px solid rgba(192,132,252,0.12);border-radius:18px;
         padding:28px 36px;margin-bottom:24px;position:relative;overflow:hidden">
        <div style="position:absolute;top:-60px;right:-60px;width:250px;height:250px;
             background:radial-gradient(circle,rgba(232,121,249,0.12) 0%,transparent 70%);
             pointer-events:none"></div>
        <div style="position:absolute;bottom:-80px;left:40%;width:200px;height:200px;
             background:radial-gradient(circle,rgba(192,132,252,0.08) 0%,transparent 70%);
             pointer-events:none"></div>
        <div style="position:absolute;top:50%;left:60%;width:300px;height:300px;
             background:radial-gradient(circle,rgba(244,114,182,0.05) 0%,transparent 70%);
             pointer-events:none"></div>
        <div style="display:inline-block;background:rgba(192,132,252,0.12);border:1px solid rgba(192,132,252,0.3);
             color:#E879F9;font-size:0.65rem;font-weight:700;letter-spacing:0.12em;
             text-transform:uppercase;padding:3px 12px;border-radius:20px;margin-bottom:12px">
            🤖 AI Powered · 5 Models Active
        </div>
        <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;
             background:linear-gradient(90deg,#E879F9 0%,#C084FC 50%,#818CF8 100%);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             line-height:1.15;margin:0 0 8px">
            🐔 Poultry Farm AI Decision Dashboard
        </div>
        <div style="font-size:0.9rem;color:#6B4F8A;font-weight:400">
            Smart analytics for profit, risk, demand &amp; contract decisions — Adjust inputs in the sidebar &amp; click Run Smart Prediction
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_kpis(res):
    risk_color  = RISK_C.get(res["risk"], C_YELLOW)
    appr_color  = APPR_C.get(res["decision"], C_YELLOW)
    profit_s    = f"₹ {res['profit']:,.0f}"
    margin_s    = f"{res['margin']:.1%} margin"
    score_s     = f"{res['score']} / 100"
    risk_icon   = {"Low":"🟢","Medium":"🟡","High":"🔴"}[res["risk"]]
    appr_icon   = {"Approve":"✅","Revise":"⚠️","Reject":"❌"}[res["decision"]]

    st.markdown(f"""
    <div class='kpi-grid'>
        <div class='kpi-card' style='--accent:linear-gradient(90deg,#EAB308,#FB923C);--glow:radial-gradient(circle,rgba(234,179,8,0.1) 0%,transparent 70%)'>
            <span class='kpi-icon'>💰</span>
            <div class='kpi-label'>Estimated Profit</div>
            <div class='kpi-value'>{profit_s}</div>
            <div class='kpi-sub'>{margin_s}</div>
        </div>
        <div class='kpi-card' style='--accent:linear-gradient(90deg,{risk_color},transparent);--glow:radial-gradient(circle,rgba(74,222,128,0.08) 0%,transparent 70%)'>
            <span class='kpi-icon'>⚠️</span>
            <div class='kpi-label'>Risk Level</div>
            <div class='kpi-value' style='font-size:1.4rem'>
                <span class='kpi-badge' style='background:rgba({",".join(str(int(risk_color.lstrip("#")[i:i+2],16)) for i in (0,2,4))},0.12);color:{risk_color};border:1px solid rgba({",".join(str(int(risk_color.lstrip("#")[i:i+2],16)) for i in (0,2,4))},0.3)'>
                    {risk_icon} {res["risk"]} Risk
                </span>
            </div>
            <div class='kpi-sub'>Composite score: {(res["mort_sc"]*0.35+res["feed_sc"]*0.25+res["rev_sc"]*0.25+res["exp_sc"]*0.15):.2f}</div>
        </div>
        <div class='kpi-card' style='--accent:linear-gradient(90deg,#A78BFA,#38BDF8);--glow:radial-gradient(circle,rgba(167,139,250,0.08) 0%,transparent 70%)'>
            <span class='kpi-icon'>🏆</span>
            <div class='kpi-label'>Performance Score</div>
            <div class='kpi-value'>{score_s}</div>
            <div class='kpi-sub'>{"Excellent" if res["score"]>=80 else "Good" if res["score"]>=65 else "Average" if res["score"]>=45 else "Poor"} performing farm</div>
        </div>
        <div class='kpi-card' style='--accent:linear-gradient(90deg,{appr_color},transparent);--glow:radial-gradient(circle,rgba(74,222,128,0.08) 0%,transparent 70%)'>
            <span class='kpi-icon'>📋</span>
            <div class='kpi-label'>Contract Decision</div>
            <div class='kpi-value' style='font-size:1.4rem'>
                <span class='kpi-badge' style='background:rgba({",".join(str(int(appr_color.lstrip("#")[i:i+2],16)) for i in (0,2,4))},0.12);color:{appr_color};border:1px solid rgba({",".join(str(int(appr_color.lstrip("#")[i:i+2],16)) for i in (0,2,4))},0.3)'>
                    {appr_icon} {res["decision"]}
                </span>
            </div>
            <div class='kpi-sub'>Confidence: {res["conf"]:.0%}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_revenue_trend(inp, res):
    np.random.seed(42)
    months = list(range(1, 13))
    base_rev = inp["revenue"] / 12
    trend_mult = 1 + (inp["roi"] / 100) / 12

    revenues = []
    for m in months:
        seasonal = 1 + 0.15 * np.sin(m * np.pi / 6)
        noise = np.random.uniform(0.9, 1.1)
        r = base_rev * (trend_mult ** m) * seasonal * noise
        revenues.append(r)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=months, y=revenues,
        marker=dict(
            color=revenues,
            colorscale=[[0, "#3B0764"], [0.25, "#6D28D9"], [0.55, "#9333EA"], [0.8, "#E879F9"], [1, "#F472B6"]],
            line=dict(color="#0D0A18", width=0.5),
        ),
        text=[f"₹{v/1000:.0f}K" for v in revenues],
        textposition="outside",
        textfont=dict(color="#94A3B8", size=10),
        hovertemplate="Month %{x}<br>Revenue: ₹%{y:,.0f}<extra></extra>",
    ))
    fig = chart_layout(fig, h=310, xlab="Month", ylab="Revenue (₹)")
    fig.update_layout(yaxis=dict(tickformat=",.0f"))
    return fig


def render_demand_forecast():
    np.random.seed(7)
    days = np.arange(0, 31)
    base = 120 + 20 * np.sin(days * 0.4) + np.random.normal(0, 8, len(days))
    demand = np.clip(base, 80, 185)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days, y=demand, mode="lines+markers",
        line=dict(color="#C084FC", width=2.5, shape="spline"),
        marker=dict(
            size=[8 if i % 5 == 0 else 4 for i in range(len(days))],
            color="#E879F9",
            line=dict(color="#0D0A18", width=2),
        ),
        fill="tozeroy",
        fillcolor="rgba(192,132,252,0.07)",
        hovertemplate="Day %{x}<br>Demand: %{y:.0f}<extra></extra>",
        name="Demand",
    ))
    fig = chart_layout(fig, h=310, xlab="Day", ylab="Demand")
    return fig


def render_risk_gauge(res):
    risk_color = RISK_C.get(res["risk"], C_YELLOW)
    comps = {
        "Mortality": round(res["mort_sc"] * 100, 1),
        "Feed Eff." : round(res["feed_sc"] * 100, 1),
        "Revenue"  : round(res["rev_sc"] * 100, 1),
        "Experience": round(res["exp_sc"] * 100, 1),
    }
    cats = list(comps.keys()) + [list(comps.keys())[0]]
    vals = list(comps.values()) + [list(comps.values())[0]]
    ri, gi, bi = (int(risk_color[1:3], 16), int(risk_color[3:5], 16), int(risk_color[5:7], 16))

    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill="toself",
        fillcolor=f"rgba({ri},{gi},{bi},0.15)",
        line=dict(color=risk_color, width=2.5),
        marker=dict(size=8, color=risk_color, line=dict(color="#0B0F19", width=2)),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 100], tickfont=dict(size=9, color=TEXT_COLOR),
                            gridcolor=GRID_COLOR, linecolor=GRID_COLOR),
            angularaxis=dict(tickfont=dict(size=12, color="#CBD5E1"), linecolor=GRID_COLOR),
        ),
        paper_bgcolor=CHART_BG, height=280,
        margin=dict(t=20, b=20, l=30, r=30),
        showlegend=False, font=dict(family=FONT, color=TEXT_COLOR),
    )
    return fig


def render_score_gauge(res):
    tc = (C_GREEN if res["score"] >= 80 else C_CYAN if res["score"] >= 65
          else C_YELLOW if res["score"] >= 45 else C_RED)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=res["score"],
        number={"font": {"color": tc, "size": 32, "family": "Syne"}},
        title={"text": "Farm Score", "font": {"color": "#94A3B8", "size": 13}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": TEXT_COLOR, "tickfont": {"color": TEXT_COLOR, "size": 9}},
            "bar":  {"color": tc, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [
                {"range": [0,  45], "color": "rgba(248,113,113,0.12)"},
                {"range": [45, 65], "color": "rgba(234,179,8,0.10)"},
                {"range": [65, 80], "color": "rgba(56,189,248,0.10)"},
                {"range": [80,100], "color": "rgba(74,222,128,0.12)"},
            ],
            "threshold": {"line": {"color": "#94A3B8", "width": 2}, "value": 65},
        },
    ))
    fig.update_layout(
        height=260, paper_bgcolor=CHART_BG,
        font=dict(family=FONT, color=TEXT_COLOR),
        margin=dict(t=30, b=10, l=20, r=20),
    )
    return fig


def render_classification_badge(res):
    score = res["score"]
    if score >= 80:
        label, icon, bg, fg = "High Performing Farm",    "🥇", "rgba(74,222,128,0.12)",  "#4ADE80"
    elif score >= 65:
        label, icon, bg, fg = "Good Performing Farm",    "🥈", "rgba(56,189,248,0.12)",  "#38BDF8"
    elif score >= 45:
        label, icon, bg, fg = "Moderate Performing Farm","⚖️", "rgba(161,132,40,0.20)",  "#EAB308"
    else:
        label, icon, bg, fg = "Low Performing Farm",     "⚠️", "rgba(248,113,113,0.12)", "#F87171"
    st.markdown(f"""
    <div class='class-badge' style='background:{bg};border:1px solid rgba(255,255,255,0.1);color:{fg}'>
        <span style='font-size:1.4rem'>{icon}</span>
        <span>{label}</span>
    </div>
    """, unsafe_allow_html=True)


def render_overview_charts(df, inp):
    # Filtered df
    fdf = df.copy()
    if inp["state"] != "All States":
        fdf = fdf[fdf["state"] == inp["state"]]
        if len(fdf) == 0:
            fdf = df.copy()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='section-title'>⚠️ Risk Distribution<span class='line'></span></div>", unsafe_allow_html=True)
        rc = fdf["risk_label"].value_counts()
        fig = go.Figure(go.Pie(
            labels=rc.index, values=rc.values, hole=0.62,
            marker=dict(colors=[RISK_C.get(r, C_YELLOW) for r in rc.index],
                        line=dict(color="#0B0F19", width=3)),
            textinfo="percent", textfont=dict(size=12, color="white"),
            hovertemplate="<b>%{label}</b><br>%{value} contracts<extra></extra>",
        ))
        fig.add_annotation(text=f"<b>{len(fdf)}</b><br>farms", x=0.5, y=0.5,
                           font=dict(size=13, color="#E2E8F0", family="Syne"), showarrow=False)
        fig.update_layout(paper_bgcolor=CHART_BG, height=260, margin=dict(t=10, b=10, l=10, r=10),
                          showlegend=True, legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center",
                                                       font=dict(color="#94A3B8", size=11),
                                                       bgcolor="rgba(0,0,0,0)"),
                          font=dict(family=FONT, color=TEXT_COLOR))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("<div class='section-title'>🤖 Decision Distribution<span class='line'></span></div>", unsafe_allow_html=True)
        ac = fdf["approval"].value_counts()
        fig = go.Figure(go.Bar(
            x=ac.index, y=ac.values,
            marker=dict(color=[APPR_C.get(a, C_YELLOW) for a in ac.index],
                        line=dict(color="#0B0F19", width=1), opacity=0.9),
            text=ac.values, textposition="outside",
            textfont=dict(color="#E2E8F0", size=14, family="Syne"),
            hovertemplate="<b>%{x}</b>: %{y}<extra></extra>",
        ))
        fig.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG, height=260,
                          margin=dict(t=10, b=10, l=10, r=10),
                          xaxis=dict(tickfont=dict(size=13, color="#CBD5E1"), gridcolor=GRID_COLOR),
                          yaxis=dict(visible=False),
                          font=dict(family=FONT, color=TEXT_COLOR))
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        st.markdown("<div class='section-title'>💰 Profit vs Revenue<span class='line'></span></div>", unsafe_allow_html=True)
        sample = fdf.sample(min(300, len(fdf)), random_state=42)
        fig = px.scatter(
            sample, x="total_revenue", y="estimated_profit",
            color="risk_label", size="farm_score",
            color_discrete_map=RISK_C,
            opacity=0.7, size_max=14,
            hover_data={"farm_id": True, "state": True},
        )
        fig.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG, height=260,
                          margin=dict(t=10, b=30, l=50, r=10),
                          xaxis=dict(gridcolor=GRID_COLOR, tickformat=",.0f",
                                     title=dict(text="Revenue (₹)", font=dict(color=TEXT_COLOR)),
                                     tickfont=dict(color=TEXT_COLOR)),
                          yaxis=dict(gridcolor=GRID_COLOR, tickformat=",.0f",
                                     title=dict(text="Profit (₹)", font=dict(color=TEXT_COLOR)),
                                     tickfont=dict(color=TEXT_COLOR)),
                          legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8", size=10),
                                      title=dict(text="", font=dict(color=TEXT_COLOR))),
                          font=dict(family=FONT, color=TEXT_COLOR))
        st.plotly_chart(fig, use_container_width=True)


def render_state_chart(df, inp):
    fdf = df.copy()
    if inp["state"] != "All States":
        fdf = fdf[fdf["state"] == inp["state"]]
        if len(fdf) == 0:
            fdf = df.copy()

    c1, c2 = st.columns([1.6, 1])
    with c1:
        st.markdown("<div class='section-title'>🗺️ Avg Profit by State (Top 15)<span class='line'></span></div>", unsafe_allow_html=True)
        sa = df.groupby("state")["estimated_profit"].mean().sort_values(ascending=True).tail(15)
        fig = go.Figure(go.Bar(
            x=sa.values, y=sa.index, orientation="h",
            marker=dict(color=sa.values,
                        colorscale=[[0, "#1E3A5F"], [0.5, "#EAB308"], [1, "#4ADE80"]],
                        showscale=False, line=dict(color="#0B0F19", width=0.5)),
            text=[f"₹{v/1000:.0f}K" for v in sa.values], textposition="outside",
            textfont=dict(color="#CBD5E1", size=11),
            hovertemplate="<b>%{y}</b><br>₹%{x:,.0f}<extra></extra>",
        ))
        fig.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG, height=360,
                          margin=dict(t=10, b=10, l=10, r=60),
                          yaxis=dict(categoryorder="total ascending",
                                     tickfont=dict(size=11, color="#CBD5E1"),
                                     gridcolor=GRID_COLOR),
                          xaxis=dict(visible=False),
                          font=dict(family=FONT, color=TEXT_COLOR))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("<div class='section-title'>🏆 Farm Score Distribution<span class='line'></span></div>", unsafe_allow_html=True)
        avg_s = fdf["farm_score"].mean()
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=fdf["farm_score"], nbinsx=26,
            marker=dict(color=C_YELLOW, opacity=0.8, line=dict(color="#0B0F19", width=0.5)),
            hovertemplate="Score %{x:.0f}: %{y} farms<extra></extra>",
        ))
        fig.add_vline(x=avg_s, line_color=C_GREEN, line_dash="dash",
                      annotation_text=f"Avg: {avg_s:.1f}",
                      annotation_font_color=C_GREEN, annotation_font_size=11)
        fig.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG, height=360,
                          margin=dict(t=10, b=30, l=50, r=10),
                          xaxis=dict(gridcolor=GRID_COLOR, title=dict(text="Farm Score", font=dict(color=TEXT_COLOR)),
                                     tickfont=dict(color=TEXT_COLOR)),
                          yaxis=dict(gridcolor=GRID_COLOR, title=dict(text="Count", font=dict(color=TEXT_COLOR)),
                                     tickfont=dict(color=TEXT_COLOR)),
                          showlegend=False, font=dict(family=FONT, color=TEXT_COLOR))
        st.plotly_chart(fig, use_container_width=True)


def render_action_items(res):
    actions_map = {
        "Approve": [
            ("✅ Issue contract immediately — all AI models approve", C_GREEN),
            ("📋 Set standard KPI monitoring cadence", C_CYAN),
            (f"💰 Expected profit: ₹{res['profit']:,.0f} ({res['margin']:.1%} margin)", C_YELLOW),
        ],
        "Revise": [
            ("📝 Request improved feed management plan from farmer", C_YELLOW),
            ("🔁 Re-evaluate after 2-week farm audit", C_ORANGE),
            (f"🎯 Target margin ≥ 10% (current: {res['margin']:.1%})", C_YELLOW),
        ],
        "Reject": [
            ("❌ Do not issue contract — risk thresholds exceeded", C_RED),
            (f"⚠️ Mortality risk: {res['mort_r']:.1%} (threshold: 22%)", C_RED),
            ("📞 Schedule farm remediation meeting before reapplication", C_ORANGE),
        ],
    }
    for text, color in actions_map.get(res["decision"], []):
        st.markdown(
            f"<div style='padding:10px 16px;background:rgba(255,255,255,0.03);border-radius:8px;"
            f"border-left:3px solid {color};margin-bottom:6px;font-size:0.85rem;color:#CBD5E1'>{text}</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    df  = load_data()
    inp = render_sidebar(df)

    # ── Header ──────────────────────────────────────────────────
    render_header()

    # ── Run button state (persist via session) ──────────────────
    if inp["run"]:
        st.session_state["last_result"] = compute_predictions(inp)
        st.session_state["last_inp"]    = inp

    has_result = "last_result" in st.session_state
    res = st.session_state.get("last_result", None)
    saved_inp = st.session_state.get("last_inp", inp)

    # ── Run Prediction Button (full width, green) ────────────────
    if not has_result:
        st.markdown("""
        <div style='text-align:center;padding:32px;background:#111827;border:1px dashed rgba(255,255,255,0.1);
             border-radius:16px;margin-bottom:20px'>
            <div style='font-size:2.4rem;margin-bottom:12px'>🚀</div>
            <div style='font-family:Syne,sans-serif;font-size:1.1rem;color:#475569;margin-bottom:8px'>
                Set your farm inputs in the sidebar
            </div>
            <div style='font-size:0.85rem;color:#334155'>
                Click <b style='color:#4ADE80'>RUN SMART PREDICTION</b> to see all analytics
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── KPI CARDS ────────────────────────────────────────────────
    if has_result:
        st.markdown("<div class='section-title'>📊 Key Business Insights<span class='line'></span></div>", unsafe_allow_html=True)
        render_kpis(res)

    # ── REVENUE PREDICTION TREND ─────────────────────────────────
    st.markdown("<div class='section-title'>💰 Revenue Prediction Trend<span class='line'></span></div>", unsafe_allow_html=True)
    with st.container():
        fig_rev = render_revenue_trend(saved_inp if has_result else inp, res if has_result else compute_predictions({
            "mortality": 5, "birds": 2000, "feed_cost": 120, "sell_price": 220,
            "farm_cap": 5000, "experience": 5, "chicks": 2000,
            "feed_total": 200000, "revenue": 800000, "duration": 12, "roi": 15,
        }))
        st.plotly_chart(fig_rev, use_container_width=True)

    # ── DEMAND FORECASTING ───────────────────────────────────────
    st.markdown("<div class='section-title'>📦 Demand Forecasting &amp; Trend Insights<span class='line'></span></div>", unsafe_allow_html=True)
    st.plotly_chart(render_demand_forecast(), use_container_width=True)

    # ── FARM PERFORMANCE CLASSIFICATION + RISK BREAKDOWN ────────
    if has_result:
        st.markdown("<div class='section-title'>🏆 Farm Performance Classification<span class='line'></span></div>", unsafe_allow_html=True)
        render_classification_badge(res)

        col_r, col_g = st.columns([1, 1])
        with col_r:
            st.markdown("<div class='section-title'>🎯 Risk Factor Radar<span class='line'></span></div>", unsafe_allow_html=True)
            st.plotly_chart(render_risk_gauge(res), use_container_width=True)
        with col_g:
            st.markdown("<div class='section-title'>🏅 Performance Score Gauge<span class='line'></span></div>", unsafe_allow_html=True)
            st.plotly_chart(render_score_gauge(res), use_container_width=True)

        # Action items
        st.markdown("<div class='section-title'>📋 AI Recommendations<span class='line'></span></div>", unsafe_allow_html=True)
        render_action_items(res)

    # ── PORTFOLIO OVERVIEW (from dataset) ────────────────────────
    if df is not None:
        st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:24px 0'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🌐 Portfolio Overview<span class='line'></span></div>", unsafe_allow_html=True)
        render_overview_charts(df, inp)

        st.markdown("<div class='section-title'>🗺️ State Performance Analysis<span class='line'></span></div>", unsafe_allow_html=True)
        render_state_chart(df, inp)

        # Heatmap
        st.markdown("<div class='section-title'>🔥 Risk Heatmap — State × Risk Level<span class='line'></span></div>", unsafe_allow_html=True)
        heat = df.groupby(["state", "risk_label"]).size().unstack(fill_value=0)
        fig_h = px.imshow(
            heat, text_auto=True, aspect="auto",
            color_continuous_scale=[[0,"#1a2436"],[0.4,"#854d0e"],[1,"#EF4444"]],
            labels=dict(color="Contracts"),
        )
        fig_h.update_layout(paper_bgcolor=CHART_BG, height=380, margin=dict(t=10,b=10,l=10,r=10),
                            font=dict(family=FONT, color=TEXT_COLOR),
                            xaxis=dict(tickfont=dict(color="#CBD5E1")),
                            yaxis=dict(tickfont=dict(color="#CBD5E1")))
        fig_h.update_traces(textfont_size=10)
        st.plotly_chart(fig_h, use_container_width=True)

        # Data Table
        with st.expander("📋 Contract Data Preview (Top 50)"):
            cols = ["farm_id","state","farm_size","contract_type","sale_type",
                    "estimated_profit","mortality_rate","farm_score","risk_label","approval"]
            v = df[cols].head(50).copy()
            v["estimated_profit"] = v["estimated_profit"].apply(lambda x: f"₹{x:,.0f}")
            v["mortality_rate"]   = v["mortality_rate"].apply(lambda x: f"{x:.2%}")
            v["farm_score"]       = v["farm_score"].round(1)
            st.dataframe(v, use_container_width=True, height=280)

    elif df is None:
        st.warning("⚠️ Dataset not found. Run: `python data/generate_dataset.py` to create sample data.")


if __name__ == "__main__":
    main()
