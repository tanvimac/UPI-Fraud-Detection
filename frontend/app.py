# frontend/app.py
"""Premium Fintech Streamlit Dashboard for UPI Fraud Detection & Transaction Monitoring.

Design & Implementation Principles:
- Technically accurate terminology (UPI Transaction Monitoring, Classified as Normal, Simulated Fraud Rate)
- Purely presentation-layer improvements — all values, charts, and metrics are calculated dynamically from actual data
- Clean dark obsidian & navy fintech aesthetic (#070B14) with elevated card surfaces (#0E1626)
- Robust pagination edge-case protection (handles total_pages <= 1 safely)
- Split-screen 'Test a Transaction' studio displaying genuine FastAPI /predict responses without fake confidence scores
- Strict separation between historical dataset monitoring and live on-demand transaction testing
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import httpx
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------
# 1. Page Configuration (Must be first Streamlit command)
# ---------------------------------------------------------------
st.set_page_config(
    page_title="UPI Transaction Monitoring & Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------
# 2. Dynamic Data Loading (Cached for performance)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    v2_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "upi_transactions_v2.csv"
    if v2_path.exists():
        return pd.read_csv(v2_path)
    csv_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "upi_transactions.csv"
    return pd.read_csv(csv_path)

@st.cache_data(show_spinner=False)
def load_processed_data() -> pd.DataFrame:
    csv_path = (
        Path(__file__).resolve().parent.parent / "data" / "processed" / "upi_transactions_features_v2.csv"
    )
    return pd.read_csv(csv_path)

raw_df = load_raw_data()
processed_df = load_processed_data()

# ---------------------------------------------------------------
# 3. Master Fintech Styling (Single Unified CSS Engine)
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Overall App Background */
    .stApp {
        background-color: #070B14 !important;
        color: #E2E8F0 !important;
    }

    /* Top Header Bar */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0B1120 !important;
        border-right: 1px solid #1E293B !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #94A3B8;
    }

    /* Radio Navigation in Sidebar */
    .stSidebar .stRadio > label {
        display: none !important;
    }
    .stSidebar .stRadio div[role="radiogroup"] {
        gap: 0.5rem;
    }
    .stSidebar .stRadio div[role="radiogroup"] > label {
        background: #111C35 !important;
        border: 1px solid #1E2D4D !important;
        border-radius: 10px !important;
        padding: 0.75rem 1rem !important;
        margin-bottom: 0.35rem !important;
        cursor: pointer !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        color: #CBD5E1 !important;
        font-weight: 500 !important;
    }
    .stSidebar .stRadio div[role="radiogroup"] > label:hover {
        background: #162444 !important;
        border-color: #00E5FF55 !important;
        color: #F8FAFC !important;
        transform: translateX(4px);
    }
    .stSidebar .stRadio div[role="radiogroup"] > label[data-checked="true"],
    .stSidebar .stRadio div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.15), rgba(16, 185, 129, 0.12)) !important;
        border: 1px solid #00E5FF !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.25) !important;
        font-weight: 700 !important;
    }

    /* Cards & Containers */
    .fintech-card {
        background: #0E1626;
        border: 1px solid #1E2D4A;
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .fintech-card:hover {
        border-color: #2D436D;
    }

    /* Metric Card Typography */
    .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8899A6;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #F8FAFC;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }

    /* Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }
    .badge-safe {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .badge-flagged {
        background: rgba(255, 51, 102, 0.15);
        color: #FF3366;
        border: 1px solid rgba(255, 51, 102, 0.35);
        box-shadow: 0 0 10px rgba(255, 51, 102, 0.2);
    }
    .badge-cyan {
        background: rgba(0, 229, 255, 0.12);
        color: #00E5FF;
        border: 1px solid rgba(0, 229, 255, 0.3);
    }

    /* Inputs, Selectboxes & Interactive Controls */
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background-color: #0B1322 !important;
        border: 1px solid #1E2D4A !important;
        border-radius: 8px !important;
        color: #F8FAFC !important;
    }
    div[data-baseweb="input"] input {
        color: #F8FAFC !important;
        font-family: inherit;
    }
    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="select"] > div:focus-within {
        border-color: #00E5FF !important;
        box-shadow: 0 0 0 1px #00E5FF !important;
    }
    .stTextInput label, .stSelectbox label, .stMultiSelect label, .stNumberInput label {
        color: #94A3B8 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00E5FF, #10B981) !important;
        color: #050B14 !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.6rem 1.4rem !important;
        box-shadow: 0 4px 14px rgba(0, 229, 255, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(0, 229, 255, 0.4) !important;
    }

    /* Dataframe wrapper styling */
    [data-testid="stDataFrame"] {
        border: 1px solid #1E2D4A;
        border-radius: 12px;
        overflow: hidden;
        background: #0B1322;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: #0E1626 !important;
        border: 1px solid #1E2D4A !important;
        border-radius: 10px !important;
        color: #F8FAFC !important;
    }
    
    code, .font-mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Custom Table Styling for Recent Transactions */
    .fintech-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        background: #0B1322;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #1E2D4A;
    }
    .fintech-table th {
        background: #111C35;
        color: #8899A6;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.9rem 1.1rem;
        text-align: left;
        border-bottom: 1px solid #1E2D4A;
    }
    .fintech-table td {
        padding: 0.85rem 1.1rem;
        color: #E2E8F0;
        font-size: 0.86rem;
        border-bottom: 1px solid #152238;
    }
    .fintech-table tr:last-child td {
        border-bottom: none;
    }
    .fintech-table tr:hover td {
        background: #121E36;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# 4. Plotly Global Dark Fintech Theme Helper
# ---------------------------------------------------------------
def format_plotly_fintech(fig: go.Figure, height: int = 340) -> go.Figure:
    """Applies high-end dark obsidian theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=20, r=20, t=45, b=25),
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#CBD5E1", size=11),
        title=dict(
            font=dict(color="#F8FAFC", size=14, family="Plus Jakarta Sans, sans-serif"),
            x=0.02,
            y=0.96,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#94A3B8", size=11),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#152238",
            zeroline=False,
            color="#64748B",
            tickfont=dict(size=10, color="#8899A6"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#152238",
            zeroline=False,
            color="#64748B",
            tickfont=dict(size=10, color="#8899A6"),
        ),
        hoverlabel=dict(
            bgcolor="#0E1626",
            bordercolor="#1E2D4A",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=12, color="#FFFFFF"),
        ),
    )
    return fig

# ---------------------------------------------------------------
# 5. UI Helper Components
# ---------------------------------------------------------------
def render_header(title: str, subtitle: str):
    """Renders a modern header with live backend status and timestamp."""
    col_t, col_s = st.columns([7, 3])
    with col_t:
        st.markdown(
            f"""
            <div style='margin-bottom: 1.25rem;'>
                <h1 style='color: #F8FAFC; margin: 0; font-size: 1.9rem; font-weight: 800; letter-spacing: -0.02em;'>
                    {title}
                </h1>
                <p style='color: #8899A6; margin: 0.25rem 0 0 0; font-size: 0.92rem;'>
                    {subtitle}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_s:
        # Check backend health
        try:
            resp = httpx.get("http://127.0.0.1:8000/health", timeout=1.5)
            if resp.status_code == 200:
                engine_status = "<span style='color:#10B981;'>●</span> Online"
                badge_class = "badge-safe"
            else:
                engine_status = "<span style='color:#FF3366;'>●</span> Error"
                badge_class = "badge-flagged"
        except Exception:
            engine_status = "<span style='color:#FF3366;'>●</span> Offline"
            badge_class = "badge-flagged"

        now_str = datetime.now().strftime("%H:%M:%S")
        st.markdown(
            f"""
            <div style='text-align: right; padding-top: 0.2rem;'>
                <div style='display: inline-flex; align-items: center; gap: 0.5rem;'>
                    <span class='badge {badge_class}' style='padding: 0.3rem 0.75rem; font-size: 0.75rem;'>
                        FastAPI Engine: {engine_status}
                    </span>
                </div>
                <div style='color: #64748B; font-size: 0.75rem; margin-top: 0.35rem; font-family: "JetBrains Mono", monospace;'>
                    Synced: {now_str}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def render_kpi(label: str, value: str, subtext: str, accent_color: str = "#00E5FF", icon: str = "📊"):
    """Renders a sleek elevated KPI card with glow border."""
    st.markdown(
        f"""
        <div class='fintech-card' style='border-top: 3px solid {accent_color};'>
            <div class='metric-label'>
                <span>{label}</span>
                <span style='font-size: 1rem;'>{icon}</span>
            </div>
            <div class='metric-value'>{value}</div>
            <div class='metric-sub'>
                <span style='color: {accent_color}; font-weight: 600;'>{subtext}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def paginate(df: pd.DataFrame, page: int, per_page: int) -> pd.DataFrame:
    """Return a slice of df for the requested page (1-based)."""
    if per_page <= 0:
        return df
    start = (page - 1) * per_page
    end = start + per_page
    return df.iloc[start:end]

# ---------------------------------------------------------------
# 6. Sidebar Navigation & Branding
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style='display: flex; align-items: center; gap: 0.75rem; padding: 0.5rem 0 1rem 0;'>
            <div style='background: linear-gradient(135deg, #00E5FF, #10B981); width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; box-shadow: 0 4px 12px rgba(0, 229, 255, 0.35);'>
                🛡️
            </div>
            <div>
                <div style='color: #F8FAFC; font-weight: 800; font-size: 1.15rem; letter-spacing: -0.01em; line-height: 1.1;'>
                    UPI GUARD
                </div>
                <div style='color: #00E5FF; font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;'>
                    UPI Transaction Monitoring
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_option = st.radio(
        "Navigation Menu",
        [
            "Dashboard Overview",
            "Transaction Monitoring",
            "Test a Transaction",
            "Fraud Analytics",
        ],
        index=0,
    )

    st.markdown("<hr style='border-color: #1E293B; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div style='background: #0E1626; border: 1px solid #1E2D4A; border-radius: 10px; padding: 0.85rem;'>
            <div style='color: #8899A6; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;'>
                Active Model Pipeline
            </div>
            <div style='color: #F8FAFC; font-size: 0.82rem; font-weight: 600; margin-top: 0.2rem;'>
                Isolation Forest V2
            </div>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-top: 0.5rem; font-size: 0.74rem; color: #64748B;'>
                <span>Dataset: V2 Synthesized</span>
                <span style='color: #10B981; font-weight: 600;'>10K Tx</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------
# 7. Page 1: Dashboard Overview
# ---------------------------------------------------------------
def page_overview():
    render_header("Executive Overview", "Historical UPI transaction dataset metrics & aggregate pattern summary")

    # Dynamic metrics calculation from actual dataset
    total = len(raw_df)
    fraud = int(raw_df["is_fraud"].sum())
    normal = total - fraud
    fraud_pct = (fraud / total * 100) if total else 0.0
    total_volume_inr = raw_df["amount"].sum()

    # 4 Modern KPI Cards with technically accurate labels
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi("Monitored Volume", f"{total:,}", f"₹{total_volume_inr:,.0f} Total Flow", accent_color="#00E5FF", icon="💳")
    with col2:
        render_kpi("Classified as Normal", f"{normal:,}", f"{100 - fraud_pct:.1f}% Dataset Share", accent_color="#10B981", icon="🛡️")
    with col3:
        render_kpi("Flagged Anomalies", f"{fraud:,}", f"{fraud_pct:.2f}% Anomaly Rate", accent_color="#FF3366", icon="🚨")
    with col4:
        render_kpi("Simulated Fraud Rate", f"{fraud_pct:.2f}%", "Baseline Contained", accent_color="#F59E0B", icon="⚡")

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    # Visualizations Row
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("<div class='fintech-card'>", unsafe_allow_html=True)
        # Class distribution donut with hollow center total
        fig_pie = px.pie(
            raw_df,
            names="is_fraud",
            title="Dataset Class Distribution",
            color="is_fraud",
            color_discrete_map={0: "#10B981", 1: "#FF3366"},
            hole=0.68,
        )
        fig_pie.update_traces(
            textinfo="percent",
            hoverinfo="label+value+percent",
            marker=dict(line=dict(color="#070B14", width=2)),
            labels=["Normal", "Fraud"],
        )
        fig_pie.update_layout(
            annotations=[
                dict(
                    text=f"<b>{total:,}</b><br><span style='font-size:10px; color:#8899A6;'>TRANSACTIONS</span>",
                    x=0.5,
                    y=0.5,
                    font_size=17,
                    showarrow=False,
                    font_color="#F8FAFC",
                    font_family="Plus Jakarta Sans, sans-serif",
                )
            ]
        )
        format_plotly_fintech(fig_pie, height=330)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chart2:
        st.markdown("<div class='fintech-card'>", unsafe_allow_html=True)
        # Density histogram
        fig_hist = px.histogram(
            raw_df,
            x="amount",
            nbins=45,
            title="Transaction Value Density (Normal vs Fraud)",
            color="is_fraud",
            marginal="rug",
            histnorm="density",
            color_discrete_map={0: "#00E5FF", 1: "#FF3366"},
            labels={"amount": "Amount (INR)", "is_fraud": "Class", "density": "Density"},
        )
        fig_hist.update_traces(opacity=0.75)
        format_plotly_fintech(fig_hist, height=330)
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Operational Insights Section (calculated dynamically)
    st.markdown(
        """
        <div style='display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;'>
            <div style='font-size: 1.1rem; font-weight: 700; color: #F8FAFC;'>Dataset Operational Insights</div>
            <div class='badge badge-cyan'>V2 Synthesized Profile</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    peak_hour = processed_df["transaction_hour"].mode().iloc[0] if "transaction_hour" in processed_df.columns and not processed_df["transaction_hour"].empty else 21
    common_merchant = raw_df["merchant_category"].value_counts().idxmax()
    common_device = raw_df["device_type"].value_counts().idxmax()
    common_location = raw_df["location"].value_counts().idxmax()

    ic1, ic2, ic3, ic4 = st.columns(4)
    with ic1:
        st.markdown(
            f"""
            <div class='fintech-card' style='padding: 0.9rem 1.1rem;'>
                <div style='color:#8899A6; font-size: 0.72rem; font-weight: 700; text-transform: uppercase;'>Peak Window</div>
                <div style='color:#F8FAFC; font-size: 1.25rem; font-weight: 700; margin-top: 0.2rem;'>{peak_hour:02d}:00 HRS</div>
                <div style='color:#00E5FF; font-size: 0.72rem;'>Max transaction density</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ic2:
        st.markdown(
            f"""
            <div class='fintech-card' style='padding: 0.9rem 1.1rem;'>
                <div style='color:#8899A6; font-size: 0.72rem; font-weight: 700; text-transform: uppercase;'>Primary Merchant</div>
                <div style='color:#F8FAFC; font-size: 1.25rem; font-weight: 700; margin-top: 0.2rem;'>{common_merchant}</div>
                <div style='color:#10B981; font-size: 0.72rem;'>Highest transaction share</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ic3:
        st.markdown(
            f"""
            <div class='fintech-card' style='padding: 0.9rem 1.1rem;'>
                <div style='color:#8899A6; font-size: 0.72rem; font-weight: 700; text-transform: uppercase;'>Dominant Device</div>
                <div style='color:#F8FAFC; font-size: 1.25rem; font-weight: 700; margin-top: 0.2rem;'>{common_device}</div>
                <div style='color:#00E5FF; font-size: 0.72rem;'>Primary client channel</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ic4:
        st.markdown(
            f"""
            <div class='fintech-card' style='padding: 0.9rem 1.1rem;'>
                <div style='color:#8899A6; font-size: 0.72rem; font-weight: 700; text-transform: uppercase;'>High-Activity Node</div>
                <div style='color:#F8FAFC; font-size: 1.25rem; font-weight: 700; margin-top: 0.2rem;'>{common_location}</div>
                <div style='color:#10B981; font-size: 0.72rem;'>Geographic volume hub</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    # Recent Transactions Ledger
    st.markdown(
        """
        <div style='display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;'>
            <div style='font-size: 1.1rem; font-weight: 700; color: #F8FAFC;'>Recent Transaction Stream</div>
            <div style='color: #64748B; font-size: 0.75rem;'>Top 5 Most Recent Dataset Ingestions</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    recent_df = raw_df.sort_values("transaction_time", ascending=False).head(5).copy()
    
    rows_html = ""
    for _, row in recent_df.iterrows():
        badge_html = (
            "<span class='badge badge-flagged'>FLAGGED FRAUD</span>"
            if row["is_fraud"] == 1
            else "<span class='badge badge-safe'>NORMAL</span>"
        )
        amt_str = f"₹{row['amount']:,.2f}"
        rows_html += (
            f"<tr>"
            f"<td class='font-mono' style='color:#00E5FF;'>{row['transaction_id']}</td>"
            f"<td style='font-weight:700;'>{amt_str}</td>"
            f"<td style='color:#8899A6;'>{row['transaction_time']}</td>"
            f"<td>{row['merchant_category']}</td>"
            f"<td>{row['device_type']}</td>"
            f"<td>{row['location']}</td>"
            f"<td>{badge_html}</td>"
            f"</tr>"
        )

    table_html = (
        f"<table class='fintech-table'>"
        f"<thead>"
        f"<tr>"
        f"<th>Transaction ID</th>"
        f"<th>Amount</th>"
        f"<th>Timestamp</th>"
        f"<th>Category</th>"
        f"<th>Device</th>"
        f"<th>Location</th>"
        f"<th>Integrity Status</th>"
        f"</tr>"
        f"</thead>"
        f"<tbody>"
        f"{rows_html}"
        f"</tbody>"
        f"</table>"
    )
    st.markdown(table_html, unsafe_allow_html=True)
    st.markdown("<div style='color:#64748B; font-size:0.75rem; margin-top:0.5rem;'>Synthetic UPI benchmark dataset — figures for demonstration only.</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# 8. Page 2: Transaction Monitoring (with Safe Pagination Fix)
# ---------------------------------------------------------------
def page_monitoring():
    render_header("Transaction Monitoring", "Filter, explore, and audit historical UPI transaction records")
    
    df = raw_df.copy()
    df["fraud_label"] = df["is_fraud"].map({0: "Normal", 1: "Fraud"})
    df["status"] = df["is_fraud"].apply(lambda x: "Fraud" if x == 1 else "Normal")

    # Filter Toolbar inside an elevated container
    with st.expander("🔍 Search & Filter Controls", expanded=True):
        col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
        with col_f1:
            tx_id_search = st.text_input("Transaction ID", placeholder="Search by ID (e.g. TXN)...", key="monitor_tx_id")
            tx_type = st.multiselect("Transaction Type", options=df["transaction_type"].unique(), key="monitor_tx_type")
        with col_f2:
            merchant = st.multiselect("Merchant Category", options=sorted(df["merchant_category"].unique()), key="monitor_merchant")
            device = st.multiselect("Device Type", options=sorted(df["device_type"].unique()), key="monitor_device")
        with col_f3:
            fraud_opt = st.selectbox("Status Filter", ["All", "Normal", "Fraud"], key="monitor_fraud_opt")
            per_page = st.selectbox("Rows per page", [10, 25, 50, 100], index=1, key="monitor_per_page")

        col_btn1, col_btn2 = st.columns([1, 5])
        with col_btn1:
            if st.button("Reset Filters", key="monitor_clear_filters"):
                st.rerun()

    # Apply filters faithfully
    if tx_id_search:
        df = df[df["transaction_id"].str.contains(tx_id_search, case=False, na=False)]
    if tx_type:
        df = df[df["transaction_type"].isin(tx_type)]
    if device:
        df = df[df["device_type"].isin(device)]
    if merchant:
        df = df[df["merchant_category"].isin(merchant)]
    if fraud_opt != "All":
        label = 1 if fraud_opt == "Fraud" else 0
        df = df[df["is_fraud"] == label]

    # Timestamp & Amount formatting
    df["transaction_time_parsed"] = pd.to_datetime(df["transaction_time"], errors="coerce", utc=True)
    df["display_time"] = df["transaction_time_parsed"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df.loc[df["display_time"].isna(), "display_time"] = df.loc[df["display_time"].isna(), "transaction_time"]
    df["display_amount"] = df["amount"].apply(lambda x: f"₹{x:,.2f}")

    display_cols = [
        "transaction_id",
        "display_amount",
        "display_time",
        "transaction_type",
        "merchant_category",
        "device_type",
        "location",
        "transactions_last_1hr",
        "status",
    ]
    df_display = df[display_cols].rename(columns={"display_amount": "amount", "display_time": "transaction_time"})

    # Safe pagination calculation
    total_records = len(df_display)
    total_pages = max(1, (total_records - 1) // per_page + 1)
    
    col_stat, col_page = st.columns([6, 4])
    with col_stat:
        st.markdown(
            f"""
            <div style='color: #8899A6; font-size: 0.85rem; padding-top: 0.5rem;'>
                Showing <strong style='color:#00E5FF;'>{min(per_page, total_records)}</strong> of <strong style='color:#F8FAFC;'>{total_records:,}</strong> matching transactions
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_page:
        # Protected against StreamlitInvalidMinMaxError when total_pages == 1
        if total_pages > 1:
            page_num = st.slider("Select Page", min_value=1, max_value=total_pages, value=1, key="monitor_page_slider")
        else:
            page_num = 1
            st.markdown(
                f"<div style='color: #64748B; font-size: 0.82rem; padding-top: 0.8rem; text-align: right;'>"
                f"Page 1 of 1"
                f"</div>",
                unsafe_allow_html=True,
            )

    paged_df = paginate(df_display, page_num, per_page)

    # Highlight fraud rows with subtle red background
    def highlight(row):
        return ["background-color: rgba(255, 51, 102, 0.12)" if row["status"] == "Fraud" else "" for _ in row]

    styled = paged_df.style.apply(highlight, axis=1)
    
    st.markdown("<div class='fintech-card' style='padding: 0.5rem;'>", unsafe_allow_html=True)
    st.dataframe(styled, use_container_width=True, height=480)
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption(f"Page {page_num} of {total_pages} • Powered by Isolation Forest V2 Features")

# ---------------------------------------------------------------
# 9. Page 3: Test a Transaction (Split-Screen AI Assessment Studio)
# ---------------------------------------------------------------
def page_test_transaction():
    render_header(
        "On-Demand Transaction Evaluation",
        "Evaluate individual transaction parameters against the deployed FastAPI Isolation Forest V2 service",
    )

    col_form, col_verdict = st.columns([5, 5])

    with col_form:
        st.markdown("<div class='fintech-card'>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style='color: #F8FAFC; font-weight: 700; font-size: 1.1rem; margin-bottom: 0.2rem;'>
                Transaction Parameters
            </div>
            <div style='color: #8899A6; font-size: 0.8rem; margin-bottom: 1rem;'>
                Enter transaction attributes to perform real-time model inference.
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form(key="test_form"):
            st.markdown("<div style='color: #00E5FF; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;'>Financial Data</div>", unsafe_allow_html=True)
            col_in1, col_in2 = st.columns(2)
            with col_in1:
                amount = st.number_input("Amount (INR)", min_value=0.01, value=1250.00, step=10.0)
            with col_in2:
                tx_time = st.text_input(
                    "Timestamp (ISO-8601)",
                    value=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                )

            st.markdown("<div style='color: #00E5FF; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; margin-top: 0.5rem;'>Contextual Attributes</div>", unsafe_allow_html=True)
            tx_type = st.selectbox("Transaction Type", ["debit", "credit"])
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                merchant = st.selectbox("Merchant Category", sorted(raw_df["merchant_category"].unique()))
                device = st.selectbox("Device Type", sorted(raw_df["device_type"].unique()))
            with col_c2:
                location = st.selectbox("Location", sorted(raw_df["location"].unique()))
                recent = st.number_input("Prior Transactions in Last 1 Hr", min_value=0, step=1, value=0)

            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
            submit = st.form_submit_button("Run Fraud Detection Analysis")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_verdict:
        st.markdown("<div class='fintech-card'>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style='color: #F8FAFC; font-weight: 700; font-size: 1.1rem; margin-bottom: 0.2rem;'>
                Model Evaluation Results
            </div>
            <div style='color: #8899A6; font-size: 0.8rem; margin-bottom: 1rem;'>
                Inference output returned directly from the FastAPI /predict endpoint.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if submit:
            payload = {
                "amount": amount,
                "transaction_time": tx_time,
                "transaction_type": tx_type,
                "merchant_category": merchant,
                "device_type": device,
                "location": location,
                "transactions_last_1hr": recent,
            }
            try:
                resp = httpx.post("http://127.0.0.1:8000/predict", json=payload, timeout=10.0)
                resp.raise_for_status()
                data = resp.json()
                prediction = data.get("prediction", "Unknown")
                message = data.get("message", "")

                if prediction == "Normal":
                    status_banner = """
                    <div style='background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(6, 78, 59, 0.3)); border: 1px solid #10B981; border-radius: 12px; padding: 1.25rem; text-align: center;'>
                        <div style='font-size: 2rem;'>🛡️</div>
                        <div style='color: #10B981; font-weight: 800; font-size: 1.4rem; letter-spacing: -0.01em;'>CLASSIFIED AS NORMAL</div>
                        <div style='color: #A7F3D0; font-size: 0.85rem; margin-top: 0.25rem;'>Consistent with normal transaction patterns</div>
                    </div>
                    """
                else:
                    status_banner = """
                    <div style='background: linear-gradient(135deg, rgba(255, 51, 102, 0.25), rgba(136, 19, 55, 0.35)); border: 1px solid #FF3366; border-radius: 12px; padding: 1.25rem; text-align: center; box-shadow: 0 0 20px rgba(255, 51, 102, 0.25);'>
                        <div style='font-size: 2rem;'>🚨</div>
                        <div style='color: #FF3366; font-weight: 800; font-size: 1.4rem; letter-spacing: -0.01em;'>FLAGGED AS SUSPICIOUS / FRAUD</div>
                        <div style='color: #FECDD3; font-size: 0.85rem; margin-top: 0.25rem;'>Anomalous transaction signature detected</div>
                    </div>
                    """

                st.markdown(status_banner, unsafe_allow_html=True)

                details_html = (
                    f"<div style='margin-top: 1.2rem; background: #0B1322; border: 1px solid #1E2D4A; border-radius: 10px; padding: 1rem;'>"
                    f"<div style='color: #8899A6; font-size: 0.72rem; font-weight: 700; text-transform: uppercase;'>API Response Message</div>"
                    f"<div style='color: #F8FAFC; font-size: 0.95rem; font-weight: 600; margin-top: 0.2rem;'>{message}</div>"
                    f"<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-top: 1rem; padding-top: 0.75rem; border-top: 1px solid #1E2D4A;'>"
                    f"<div><span style='color: #64748B; font-size: 0.75rem;'>Amount Evaluated:</span><div style='color: #00E5FF; font-weight: 700;'>₹{amount:,.2f}</div></div>"
                    f"<div><span style='color: #64748B; font-size: 0.75rem;'>Device Channel:</span><div style='color: #F8FAFC; font-weight: 600;'>{device}</div></div>"
                    f"<div><span style='color: #64748B; font-size: 0.75rem;'>Prior 1-Hr Transactions:</span><div style='color: #F8FAFC; font-weight: 600;'>{recent} prior tx</div></div>"
                    f"<div><span style='color: #64748B; font-size: 0.75rem;'>Model Version:</span><div style='color: #10B981; font-weight: 600;'>IsolationForest-V2</div></div>"
                    f"</div>"
                    f"</div>"
                )
                st.markdown(details_html, unsafe_allow_html=True)

            except Exception as e:
                st.error("FastAPI Backend Communication Failure")
                st.caption(f"Endpoint: http://127.0.0.1:8000/predict. Details: {e}")
        else:
            st.markdown(
                """
                <div style='background: #0B1322; border: 1px dashed #1E2D4A; border-radius: 12px; padding: 3rem 1.5rem; text-align: center;'>
                    <div style='font-size: 2.2rem; opacity: 0.5;'>⚡</div>
                    <div style='color: #F8FAFC; font-weight: 600; font-size: 1rem; margin-top: 0.5rem;'>
                        Awaiting Submission
                    </div>
                    <div style='color: #64748B; font-size: 0.8rem; margin-top: 0.25rem; max-width: 320px; margin-left: auto; margin-right: auto;'>
                        Fill in transaction parameters on the left panel and click <strong>'Run Fraud Detection Analysis'</strong> to trigger live ML scoring.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# 10. Page 4: Fraud Analytics (Balanced 2x2 Grid + Density Plot)
# ---------------------------------------------------------------
def page_fraud_analytics():
    render_header("Fraud Analytics & Patterns", "Macro-level inspection of simulated anomaly distributions")

    fraud_df = raw_df[raw_df["is_fraud"] == 1]

    # Row 1: Merchant & Device
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div class='fintech-card'>", unsafe_allow_html=True)
        m_counts = fraud_df["merchant_category"].value_counts().reset_index(name="count")
        m_counts.columns = ["merchant_category", "count"]
        fig1 = px.bar(
            m_counts,
            x="merchant_category",
            y="count",
            title="Fraud Incidence by Merchant Category",
            color="count",
            color_continuous_scale=["#00E5FF", "#10B981"],
        )
        fig1.update_coloraxes(showscale=False)
        format_plotly_fintech(fig1, height=290)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='fintech-card'>", unsafe_allow_html=True)
        d_counts = fraud_df["device_type"].value_counts().reset_index(name="count")
        d_counts.columns = ["device_type", "count"]
        fig2 = px.bar(
            d_counts,
            x="device_type",
            y="count",
            title="Fraud Distribution Across Device Channels",
            color="count",
            color_continuous_scale=["#38BDF8", "#FF3366"],
        )
        fig2.update_coloraxes(showscale=False)
        format_plotly_fintech(fig2, height=290)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    # Row 2: Location & Hourly Temporal Pattern
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("<div class='fintech-card'>", unsafe_allow_html=True)
        l_counts = fraud_df["location"].value_counts().reset_index(name="count")
        l_counts.columns = ["location", "count"]
        fig3 = px.bar(
            l_counts,
            x="location",
            y="count",
            title="Geographic Anomaly Hotspots",
            color="count",
            color_continuous_scale=["#6366F1", "#A855F7"],
        )
        fig3.update_coloraxes(showscale=False)
        format_plotly_fintech(fig3, height=290)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_d:
        st.markdown("<div class='fintech-card'>", unsafe_allow_html=True)
        if "transaction_hour" in processed_df.columns:
            hour_counts = (
                processed_df[processed_df["is_fraud"] == 1]
                .groupby("transaction_hour")
                .size()
                .reset_index(name="count")
            )
        else:
            raw_df["hour"] = pd.to_datetime(raw_df["transaction_time"], errors="coerce").dt.hour
            hour_counts = (
                raw_df[raw_df["is_fraud"] == 1]
                .groupby("hour")
                .size()
                .reset_index(name="count")
                .rename(columns={"hour": "transaction_hour"})
            )
        fig4 = px.bar(
            hour_counts,
            x="transaction_hour",
            y="count",
            title="Temporal Threat Frequency (Hour of Day)",
            labels={"transaction_hour": "Hour (0-23)", "count": "Flagged Count"},
            color="count",
            color_continuous_scale=["#00E5FF", "#FF3366"],
        )
        fig4.update_coloraxes(showscale=False)
        format_plotly_fintech(fig4, height=290)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    # Full Width Row: Value Density Spectrum
    st.markdown("<div class='fintech-card'>", unsafe_allow_html=True)
    fig5 = px.histogram(
        fraud_df,
        x="amount",
        nbins=40,
        histnorm="density",
        title="Fraud Transaction Value Spectrum (Density Distribution)",
        color_discrete_sequence=["#FF3366"],
    )
    format_plotly_fintech(fig5, height=280)
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption("All analytics charts derived dynamically from synthetic V2 UPI transaction records.")

# ---------------------------------------------------------------
# 11. Router
# ---------------------------------------------------------------
if nav_option == "Dashboard Overview":
    page_overview()
elif nav_option == "Transaction Monitoring":
    page_monitoring()
elif nav_option == "Test a Transaction":
    page_test_transaction()
elif nav_option == "Fraud Analytics":
    page_fraud_analytics()
