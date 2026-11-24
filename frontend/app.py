# frontend/app.py
"""Modern fintech Streamlit dashboard for UPI Fraud Detection.

Features:
- Dark navy sidebar with product branding.
- Light grey main area.
- White rounded KPI cards.
- Consistent page headings and subtitles.
- Clean filters and tables.
- FastAPI integration for live prediction.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import httpx
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------
# Page configuration (must be called before any Streamlit commands)
# ---------------------------------------------------------------
st.set_page_config(page_title="UPI Fraud Detection Dashboard", layout="wide")

# ---------------------------------------------------------------
# Load data (cached for performance)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    csv_path = (
        Path(__file__).resolve().parent.parent / "data" / "raw" / "upi_transactions.csv"
    )
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
# Custom CSS – targets generic Streamlit selectors; works across versions
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Backgrounds */
    .stApp {background-color: #081329; color: #F4F7FF;}
    [data-testid='stSidebar'] {background-color: #071A38; color: #F4F7FF;}
    [data-testid='stSidebar'] .css-1d391kg, [data-testid='stSidebar'] .css-1d391kg * {color: #F4F7FF;}
    /* Card */
    .custom-card {
        background: #122344;
        border: 1px solid #203C66;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        backdrop-filter: blur(4px);
        margin-bottom: 1rem;
    }
    .custom-card h4 {margin: 0; color: #16D6C1; font-size: 1.2rem;}
    .custom-card .value {font-size: 1.6rem; font-weight: bold; color: #F4F7FF;}
    /* Sidebar active */
    .stSidebar .stRadio > div > label[data-baseweb=radio] input:checked + div {
        background-color: #16D6C1 !important;}
    /* Badges */
    .badge {padding: 0.2rem 0.6rem; border-radius: 0.5rem; font-weight: 600;}
    .badge-normal {background-color: #19C784; color: #081329;}
    .badge-fraud {background-color: #FF5268; color: #081329;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Helper function to render a card
# ---------------------------------------------------------------
def card(title: str, value: str, colour: str = "#101D32"):
    """Render a white card with a coloured title.
    Used for KPI cards and prediction result panels.
    """
    st.markdown(
        f"""
        <div class='custom-card'>
          <h4 style='color:{colour}'>{title}</h4>
          <div class='value'>{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------
# Sidebar – branding + navigation (radio)
# ---------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style='display:flex; align-items:center; gap:0.5rem; color:#ffffff;'>
      <span style='font-size:1.5rem;'>🛡️</span>
      <strong>UPI Guard</strong>
    </div>
    <hr style='border-color:#ffffff33'>
    """,
    unsafe_allow_html=True,
)

nav_option = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Transaction Monitoring",
        "Test a Transaction",
        "Fraud Analytics",
    ],
    index=0,
)

st.sidebar.markdown("---\n*Synthetic Data Demo*", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Helper: consistent page header
# ---------------------------------------------------------------
def page_header(title: str, subtitle: str = ""):
    st.header(title)
    if subtitle:
        st.subheader(subtitle)

# ---------------------------------------------------------------
# Page: Dashboard Overview
# ---------------------------------------------------------------
def page_overview():
    """Premium Dashboard Overview – redesigned for a fintech analytics look.
    Uses the V2 dataset to compute KPIs, insights, and recent transactions.
    """
    # Basic metrics
    total = len(raw_df)
    fraud = raw_df["is_fraud"].sum()
    normal = total - fraud
    fraud_pct = fraud / total * 100 if total else 0

    # Header with title, subtitle, backend status, and timestamp
    col_title, col_status = st.columns([8, 2])
    with col_title:
        st.markdown("""
        <h1 style='color:#F4F7FF; margin:0;'>UPI Fraud Detection System</h1>
        <p style='color:#B7C5DC; margin:0;'>Real-time transaction monitoring and fraud analytics</p>
        """, unsafe_allow_html=True)
    with col_status:
        # Backend connection status (lightweight health check)
        try:
            resp = httpx.get("http://127.0.0.1:8000/health", timeout=2.0)
            if resp.status_code == 200:
                status_msg = "✅ Backend online"
                status_colour = "#19C784"
            else:
                status_msg = "⚠️ Backend error"
                status_colour = "#FF5268"
        except Exception:
            status_msg = "❌ Backend offline"
            status_colour = "#FF5268"
        st.markdown(f"<div style='color:{status_colour}; font-weight:600;'>{status_msg}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:#B7C5DC; font-size:0.9rem;'>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>", unsafe_allow_html=True)

    st.markdown("---")

    # -------------------- KPI cards --------------------
    st.markdown("<div style='display:flex; gap:1rem; flex-wrap:wrap;'>", unsafe_allow_html=True)
    def kpi_card(title, value, colour="#16D6C1"):
        st.markdown(
            f"""
            <div class='custom-card' style='flex:1; min-width:200px;'>
              <h4 style='color:{colour}; margin:0;'>{title}</h4>
              <div class='value'>{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    kpi_card("Total Transactions", f"{total:,}")
    kpi_card("Normal Transactions", f"{normal:,}", colour="#16A085")
    kpi_card("Fraud Transactions", f"{fraud:,}", colour="#E74C3C")
    kpi_card("Fraud Rate", f"{fraud_pct:.2f}%", colour="#E74C3C")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>")

    # -------------------- Analytics charts --------------------
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        # Donut chart for class distribution
        fig = px.pie(
            raw_df,
            names="is_fraud",
            title="Normal vs Fraud Distribution",
            color_discrete_map={0: "#16A085", 1: "#E74C3C"},
            hole=0.5,
        )
        fig.update_traces(textinfo="percent+label", hoverinfo="label+value+percent")
        # Show total count in the center using annotation
        fig.update_layout(
            annotations=[dict(text=f"{total:,}", x=0.5, y=0.5, font_size=20, showarrow=False, font_color="#F4F7FF")],
            legend=dict(font=dict(color="#F4F7FF")),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col_chart2:
        # Histogram of transaction amounts
        fig2 = px.histogram(
            raw_df,
            x="amount",
            nbins=50,
            title="Transaction Amount Distribution",
            color="is_fraud",
            marginal="rug",
            histnorm="density",
            color_discrete_map={0: "#16A085", 1: "#E74C3C"},
        )
        fig2.update_layout(
            xaxis_title="Amount (INR)",
            yaxis_title="Density",
            legend=dict(font=dict(color="#F4F7FF")),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # -------------------- Insights section --------------------
    st.markdown("## Insights")
    # Compute insight metrics
    peak_hour = processed_df["transaction_hour"].mode().iloc[0] if not processed_df["transaction_hour"].empty else None
    common_merchant = raw_df["merchant_category"].value_counts().idxmax()
    common_device = raw_df["device_type"].value_counts().idxmax()
    common_location = raw_df["location"].value_counts().idxmax()

    # Insight cards – same style as KPI but smaller
    st.markdown("<div style='display:flex; gap:1rem; flex-wrap:wrap;'>", unsafe_allow_html=True)
    def insight_card(title, value):
        st.markdown(
            f"""
            <div class='custom-card' style='flex:1; min-width:200px; background:#122344; border:1px solid #203C66;'>
              <h4 style='color:#16D6C1; margin:0;'>{title}</h4>
              <div class='value' style='color:#F4F7FF;'>{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    insight_card("Peak Transaction Hour", f"{peak_hour}:00")
    insight_card("Top Merchant Category", common_merchant)
    insight_card("Top Device Type", common_device)
    insight_card("Top Location", common_location)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>")

    # -------------------- Recent Transactions --------------------
    st.markdown("## Recent Transactions")
    recent_df = raw_df.sort_values("transaction_time", ascending=False).head(5).copy()
    # Add status badge
    recent_df["status"] = recent_df["is_fraud"].apply(lambda x: f"<span class='badge badge-fraud'>Fraud</span>" if x == 1 else f"<span class='badge badge-normal'>Normal</span>")
    display_cols = ["transaction_id", "amount", "transaction_time", "merchant_category", "device_type", "location", "status"]
    recent_display = recent_df[display_cols]
    # Format amount and time for readability
    recent_display["amount"] = recent_display["amount"].apply(lambda x: f"₹{x:,.2f}")
    # Use st.dataframe with HTML rendering disabled – we will use st.markdown for each row
    # Convert to HTML table preserving badges
    html = recent_display.to_html(index=False, escape=False)
    st.markdown(html, unsafe_allow_html=True)

    st.caption("Data are synthetic – figures do not reflect real UPI usage.")
    total = len(raw_df)
    fraud = raw_df["is_fraud"].sum()  # column name assumed
    normal = total - fraud
    fraud_pct = fraud / total * 100

    page_header("📊 Dashboard Overview", "Key performance indicators")
    # KPI cards – 4 columns on desktop
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        card("Total Transactions", f"{total:,}")
    with col2:
        card("Normal Transactions", f"{normal:,}", colour="#16A085")
    with col3:
        card("Fraud Transactions", f"{fraud:,}", colour="#E74C3C")
    with col4:
        card("Fraud %", f"{fraud_pct:.2f}%", colour="#E74C3C")

    # Donut chart for class distribution
    fig = px.pie(
        raw_df,
        names="is_fraud",
        title="Transaction class distribution",
        color_discrete_map={0: "#16A085", 1: "#E74C3C"},
        hole=0.5,
    )
    fig.update_traces(textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)

    # Amount distribution (density histogram)
    fig2 = px.histogram(
        raw_df,
        x="amount",
        color="is_fraud",
        nbins=50,
        marginal="rug",
        histnorm="density",
        color_discrete_map={0: "#16A085", 1: "#E74C3C"},
        title="Transaction amount distribution",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.caption("Data are synthetic – percentages do not reflect real UPI usage.")

# ---------------------------------------------------------------
# Helper: consistent page header with status and timestamp
# ---------------------------------------------------------------
def page_header_status(title: str, subtitle: str = ""):
    col1, col2 = st.columns([8, 2])
    with col1:
        st.markdown(f"<h1 style='color:#FFFFFF; margin:0;'>{title}</h1>", unsafe_allow_html=True)
        if subtitle:
            st.markdown(f"<p style='color:#B7C5DC; margin:0;'>{subtitle}</p>", unsafe_allow_html=True)
    with col2:
        st.markdown(
            f"""
            <div style='display:flex; align-items:center; gap:0.3rem;'>
                <span style='color:#19C784;'>&#x25CF;</span>
                <span style='color:#B7C5DC; font-size:0.9rem;'>Online</span>
            </div>
            <div style='color:#B7C5DC; font-size:0.8rem;'>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------
# Pagination helper
# ---------------------------------------------------------------
def paginate(df: pd.DataFrame, page: int, per_page: int) -> pd.DataFrame:
    """Return a slice of df for the requested page (1‑based)."""
    if per_page <= 0:
        return df
    start = (page - 1) * per_page
    end = start + per_page
    return df.iloc[start:end]

# ---------------------------------------------------------------
# Updated CSS – include inputs, buttons, and tooltip readability
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {background-color: #081329; color: #EAF1FF;}
    [data-testid='stSidebar'] {background-color: #071A38; color: #F4F7FF;}
    [data-testid='stSidebar'] .css-1d391kg, [data-testid='stSidebar'] .css-1d391kg * {color: #F4F7FF;}
    .custom-card {background: #122344; border: 1px solid #203C66; border-radius: 12px; padding: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.4); margin-bottom: 1rem;}
    .custom-card h4 {margin: 0; color: #16D6C1; font-size: 1.2rem;}
    .custom-card .value {font-size: 1.6rem; font-weight: bold; color: #F4F7FF;}
    .stSidebar .stRadio > div > label[data-baseweb=radio] input:checked + div {background-color: #16D6C1 !important;}
    .badge {padding: 0.2rem 0.6rem; border-radius: 0.5rem; font-weight: 600;}
    .badge-normal {background-color: #19C784; color: #081329;}
    .badge-fraud {background-color: #FF5268; color: #081329;}
    .stTextInput > div > div > input {background-color: #0D1B35 !important; color: #FFFFFF !important; border: 1px solid #385477 !important;}
    .stSelectbox > div > div > div {background-color: #0D1B35 !important; color: #FFFFFF !important; border: 1px solid #385477 !important;}
    .stNumberInput > div > div > input {background-color: #0D1B35 !important; color: #FFFFFF !important; border: 1px solid #385477 !important;}
    input::placeholder {color: #9BAAC5 !important; opacity:1;}
    .stButton > button {background: linear-gradient(135deg, #16D6C1, #18BFEA) !important; color: #081329 !important; font-weight: 600; border: none;}
    .stButton > button:hover {opacity:0.9;}
    .plotly .svg {font-family: Arial, sans-serif; color: #F4F7FF;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Page: Transaction Monitoring – updated for pagination & badges
# ---------------------------------------------------------------
def page_monitoring():
    page_header_status("🔎 Transaction Monitoring", "Filter and explore transactions")
    df = raw_df.copy()
    # NOTE: Duplicate filter and pagination blocks removed to avoid StreamlitDuplicateElementId errors.
    # The original implementation above (lines 383-444) provides the full functionality.
    # No further duplicated sections are needed.

    df["fraud_label"] = df["is_fraud"].map({0: "Normal", 1: "Fraud"})
    df["status"] = df["is_fraud"].apply(lambda x: "Fraud" if x == 1 else "Normal")

    # ----- Filters -----
    with st.expander("Filters", expanded=True):
        col_a, col_b = st.columns([2, 2])
        with col_a:
            tx_id_search = st.text_input("Transaction ID", key="monitor_tx_id")
            tx_type = st.multiselect("Transaction type", options=df["transaction_type"].unique(), key="monitor_tx_type")
            merchant = st.multiselect("Merchant category", options=df["merchant_category"].unique(), key="monitor_merchant")
        with col_b:
            device = st.multiselect("Device type", options=df["device_type"].unique(), key="monitor_device")
            fraud_opt = st.selectbox("Fraud label", ["All", "Normal", "Fraud"], key="monitor_fraud_opt")
        if st.button("Clear filters", key="monitor_clear_filters"):
            st.rerun()

    # Apply filters
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

    # Timestamp handling
    df["transaction_time_parsed"] = pd.to_datetime(df["transaction_time"], errors="coerce", utc=True)
    df["display_time"] = df["transaction_time_parsed"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df.loc[df["display_time"].isna(), "display_time"] = df.loc[df["display_time"].isna(), "transaction_time"]

    # Amount formatting
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

    # Highlight fraud rows with a subtle dark red background
    def highlight(row):
        return ["background-color:#2A0A0F" if row["status"] == "Fraud" else "" for _ in row]

    per_page = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)
    total_pages = max(1, (len(df_display) - 1) // per_page + 1)
    page_num = st.slider("Page", 1, total_pages, 1)

    paged_df = paginate(df_display, page_num, per_page)
    styled = paged_df.style.apply(highlight, axis=1)
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.dataframe(styled, use_container_width=True, height=500)
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption(f"Showing {len(paged_df)} of {len(df_display)} filtered transactions")



# ---------------------------------------------------------------
# Page: Test a Transaction
# ---------------------------------------------------------------
def page_test_transaction():
    page_header_status("🧪 Test a Transaction", "Submit a transaction and get a prediction")
    with st.form(key="test_form"):
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Amount (INR)", min_value=0.01, step=0.01)
            tx_time = st.text_input(
                "Transaction time (ISO‑8601)",
                value=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            )
        with col2:
            tx_type = st.selectbox("Transaction type", ["debit", "credit"])
            merchant = st.selectbox("Merchant category", sorted(raw_df["merchant_category"].unique()))
            device = st.selectbox("Device type", sorted(raw_df["device_type"].unique()))
            location = st.selectbox("Location", sorted(raw_df["location"].unique()))
            recent = st.number_input("Transactions in the last hour", min_value=0, step=1, value=0)
        submit = st.form_submit_button("Analyze Transaction")

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
            colour = "#16A085" if data["prediction"] == "Normal" else "#E74C3C"
            card("Prediction", data["prediction"], colour=colour)
            st.success(data.get("message", ""))
        except Exception as e:
            st.error("Could not reach the FastAPI backend. Is it running?")
            st.caption(str(e))

# ---------------------------------------------------------------
# Page: Fraud Analytics
# ---------------------------------------------------------------
def page_fraud_analytics():
    page_header_status("📈 Fraud Analytics", "Deep dive into simulated fraud patterns")
    fraud_df = raw_df[raw_df["is_fraud"] == 1]
    col_left, col_right = st.columns(2)
    with col_left:
        fig1 = px.bar(
            fraud_df["merchant_category"].value_counts().reset_index(name="count"),
            x="merchant_category",
            y="count",
            labels={"merchant_category": "Merchant Category", "count": "Count"},
            title="Fraud by Merchant Category",
        )
        st.plotly_chart(fig1, use_container_width=True)
        fig2 = px.bar(
            fraud_df["device_type"].value_counts().reset_index(name="count").rename(columns={"index": "device_type"}),
            x="device_type",
            y="count",
            labels={"device_type": "Device Type", "count": "Count"},
            title="Fraud by Device Type",
        )
        st.plotly_chart(fig2, use_container_width=True)
    with col_right:
        fig3 = px.bar(
            fraud_df["location"].value_counts().reset_index(name="count").rename(columns={"index": "location"}),
            x="location",
            y="count",
            labels={"location": "Location", "count": "Count"},
            title="Fraud by Location",
        )
        st.plotly_chart(fig3, use_container_width=True)
        if "transaction_hour" in processed_df.columns:
            hour_counts = (
                processed_df[processed_df["is_fraud"] == 1]
                .groupby("transaction_hour")
                .size()
                .reset_index(name="count")
            )
        else:
            raw_df["hour"] = pd.to_datetime(raw_df["transaction_time"], format="%Y-%m-%dT%H:%M:%S", errors="coerce").dt.hour
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
            title="Fraud by Hour of Day",
            labels={"transaction_hour": "Hour (0‑23)", "count": "Count"},
        )
        st.plotly_chart(fig4, use_container_width=True)
        fig5 = px.histogram(
            fraud_df,
            x="amount",
            nbins=30,
            histnorm="density",
            title="Amount distribution for fraudulent transactions",
        )
        st.plotly_chart(fig5, use_container_width=True)
        st.caption("All charts use the synthetic dataset generated for this demo.")

# ---------------------------------------------------------------
# Render the selected page
# ---------------------------------------------------------------
if nav_option == "Dashboard Overview":
    page_overview()
elif nav_option == "Transaction Monitoring":
    page_monitoring()
elif nav_option == "Test a Transaction":
    page_test_transaction()
elif nav_option == "Fraud Analytics":
    page_fraud_analytics()
