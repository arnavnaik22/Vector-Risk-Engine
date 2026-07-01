import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import importlib
import html

Groq = None
try:
    Groq = importlib.import_module("groq").Groq
except Exception:
    Groq = None

# CONFIGURATION & AUTHENTICATION
st.set_page_config(
    page_title="Behavioural Intelligence Engine",
    layout="wide",
    page_icon="📈"
)

# API Key Validation
API_KEY = st.secrets.get("GROQ_API_KEY") if "GROQ_API_KEY" in st.secrets else None
client = Groq(api_key=API_KEY) if API_KEY else None

if client is None:
    st.warning("Groq API key not configured. Strategy generation will use a local fallback.")

st.markdown(
    """
    <style>
        .dashboard-badge {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        .dashboard-card {
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .dashboard-card h4 {
            margin: 0 0 0.4rem 0;
            font-size: 0.92rem;
            color: #334155;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .dashboard-card p {
            margin: 0;
            color: #111827;
            line-height: 1.5;
        }
        .footer-note {
            margin-top: 1rem;
            padding-top: 0.75rem;
            border-top: 1px solid rgba(15, 23, 42, 0.08);
            color: #64748b;
            font-size: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Path Configurations
BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent

DATA_PATH = ROOT / "data" / "processed" / "labeled_customers.csv"
MODEL_CLASS_PATH = ROOT / "models" / "xgb_classifier.json"
MODEL_REG_PATH = ROOT / "models" / "xgb_regressor.json"

# RESOURCE MANAGEMENT (CACHED)
@st.cache_data
def load_data():
    """Loads and caches the customer dataset."""
    if not DATA_PATH.exists():
        st.error(f"Data file not found at: {DATA_PATH}")
        st.stop()
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    """Loads trained XGBoost models from disk."""
    if not MODEL_CLASS_PATH.exists() or not MODEL_REG_PATH.exists():
        st.error("Model files missing. Please run the training script first.")
        st.stop()
        
    clf = xgb.XGBClassifier()
    clf.load_model(MODEL_CLASS_PATH)
    
    reg = xgb.XGBRegressor()
    reg.load_model(MODEL_REG_PATH)
    
    return clf, reg

@st.cache_resource
def load_explainers(_clf_model, _reg_model):
    """
    Initializes SHAP explainers.
    Cached to prevent re-initialization on every user interaction (Performance Fix).
    """
    exp_clf = shap.TreeExplainer(_clf_model)
    exp_reg = shap.TreeExplainer(_reg_model)
    return exp_clf, exp_reg

# Initialize Application State
df = load_data()
if 'Unnamed: 0' in df.columns: df = df.drop(columns=['Unnamed: 0'])

clf_model, reg_model = load_models()
explainer_clf, explainer_reg = load_explainers(clf_model, reg_model)

# Feature Definition
FEATURES = ['Recency', 'Frequency', 'Avg_Monthly_Spend', 'Tenure', 'AOV', 'Velocity_Recent', 'Velocity_Drift', 'Return_Count']
X = df[FEATURES]

# Pre-compute Global Context (Percentiles)
all_preds_log = reg_model.predict(X)
df['Predicted_Spend'] = np.expm1(all_preds_log)
df['Value_Percentile'] = df['Predicted_Spend'].rank(pct=True)

# BUSINESS LOGIC LAYER
def calculate_roi_strategy(segment_prob, potential_spend, segment_name):
    """Calculates ROI and determines the optimal engagement strategy."""
    GROSS_MARGIN = 0.40
    potential_profit = potential_spend * GROSS_MARGIN
    
    if segment_name == "At-Risk":
        cost = 8.0
        captured_value = potential_profit * 0.40
        strategy = "DEFEND"
    elif segment_name == "High Value":
        cost = 3.0
        captured_value = potential_profit * 0.20
        strategy = "UPSELL"
    else:
        cost = 5.0
        captured_value = potential_profit * 0.25
        strategy = "NURTURE"

    roi = captured_value - cost
    
    if roi > 0:
        return f"{strategy}", f"Est. ROI: +${roi:.0f} (Cost: ${cost})"
    elif roi > -2:
        return "MONITOR", f"Est. ROI: ${roi:.0f} (Cost: ${cost})"
    return "SKIP", f"Value (${captured_value:.0f}) < Cost (${cost})"


def segment_badge(segment_name):
    if segment_name == "High Value":
        return "#1f7a3a", "#e8f5eb"
    if segment_name == "At-Risk":
        return "#b42318", "#fdecec"
    return "#b78103", "#fff4d6"


def action_badge(action_name):
    if action_name == "DEFEND":
        return "#b42318", "#fdecec"
    if action_name == "UPSELL":
        return "#1f7a3a", "#e8f5eb"
    if action_name == "NURTURE":
        return "#0369a1", "#e6f4fb"
    if action_name == "MONITOR":
        return "#6b7280", "#f1f5f9"
    return "#6b7280", "#f1f5f9"


def render_badge(label, fg, bg):
    st.markdown(
        f'<span class="dashboard-badge" style="color:{fg}; background:{bg};">{html.escape(label)}</span>',
        unsafe_allow_html=True,
    )


def render_strategy_output(raw_text):
    sections = {"THE FINANCIALS": "", "THE SIGNAL": "", "THE TACTIC": ""}
    current = None

    for line in raw_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("1. THE FINANCIALS:"):
            current = "THE FINANCIALS"
            sections[current] += stripped.split(":", 1)[1].strip() + "\n"
        elif stripped.startswith("2. THE SIGNAL:"):
            current = "THE SIGNAL"
            sections[current] += stripped.split(":", 1)[1].strip() + "\n"
        elif stripped.startswith("3. THE TACTIC:"):
            current = "THE TACTIC"
            sections[current] += stripped.split(":", 1)[1].strip() + "\n"
        elif current:
            sections[current] += line + "\n"

    if any(value.strip() for value in sections.values()):
        for title in ["THE FINANCIALS", "THE SIGNAL", "THE TACTIC"]:
            body = sections[title].strip()
            st.markdown(
                f'''
                <div class="dashboard-card">
                    <h4>{title}</h4>
                    <p>{html.escape(body).replace("\n", "<br>")}</p>
                </div>
                ''',
                unsafe_allow_html=True,
            )
            st.write("")
    else:
        st.markdown(
            f'<div class="dashboard-card"><p>{html.escape(raw_text).replace("\n", "<br>")}</p></div>',
            unsafe_allow_html=True,
        )

def generate_strategic_narrative(context_data):
    """Uses LLM to generate a qualitative strategic report."""
    if client is None:
        return (
            "Local fallback: the customer is in the "
            f"{context_data['segment']} segment with {context_data['insight']} "
            f"and {context_data['roi_action']}. "
            "Configure GROQ_API_KEY in Streamlit Cloud secrets to enable LLM narratives."
        )

    prompt = f"""
    Role: Senior Retention Strategist.
    
    Client Profile:
    - ID: {context_data['id']}
    - Segment: {context_data['segment']} (Conf: {context_data['prob']})
    - Future Value: {context_data['spend']} ({context_data['percentile']})
    - Momentum: {context_data['insight']}
    - Action: {context_data['roi_action']}

    Important:
    The dashboard insight is heuristic.

    If the SHAP drivers indicate stronger evidence than the dashboard insight, prioritize the SHAP evidence.
    Explain any conflicts explicitly.
    
    Key Drivers:
    - Risk: {context_data['risk_drivers']}
    - Spend: {context_data['spend_drivers']}
    
    Playbook:
    - Low AOV -> Bundle/Upsell.
    - Low Frequency -> Loyalty/Subscription.
    - Low Recency -> Win-back.
    - High Velocity -> Flash Sale.
    
    Constraints:
    - Use "USD" instead of currency symbols.
    - Be concise and tactical.
    
    Output Format:
    1. THE FINANCIALS: ROI justification.
    2. THE SIGNAL: Behavioral interpretation (conflicts included).
    3. THE TACTIC: Specific campaign recommendation.
    """
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.5
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating strategy: {e}"

# FRONTEND / DASHBOARD
st.title("Behavioral Intelligence Engine")
st.markdown("### Production-Grade Customer Decision System")

# --- SIDEBAR ---
st.sidebar.header("Controls")
customer_ids = df['CustomerID'].unique()
selected_id = st.sidebar.selectbox("Select Customer ID", customer_ids)
customer_idx = df[df['CustomerID'] == selected_id].index[0]
customer_row = df.iloc[customer_idx]

# --- KEY METRICS ---
col1, col2, col3, col4 = st.columns(4)

# Inference
pred_cluster = clf_model.predict(X.iloc[[customer_idx]])[0]
probs = clf_model.predict_proba(X.iloc[[customer_idx]])[0]
segment_prob = probs[pred_cluster]

cluster_map = {0: 'At-Risk', 1: 'Average', 2: 'High Value'}
cluster_name = cluster_map.get(pred_cluster, "Unknown")

log_spend = reg_model.predict(X.iloc[[customer_idx]])[0]
real_spend = np.expm1(log_spend) if log_spend > 0 else 0

value_pct = df.loc[customer_idx, 'Value_Percentile']

# Display Metrics
col1.metric("Model Segment", cluster_name)
col2.metric("Model Confidence", f"{segment_prob:.1%}")
col3.metric("Predicted 90-Day Revenue", f"USD {real_spend:.0f}")

if value_pct >= 0.9: 
    tier, color = "Top 10%", "normal"
elif value_pct >= 0.7: 
    tier, color = "Top 30%", "normal"
elif value_pct >= 0.5: 
    tier, color = "Mid Tier", "off"
else: 
    tier, color = "Low Priority", "off"
col4.metric("Value Tier", tier, f"{value_pct:.0%} Percentile", delta_color=color)

st.progress(float(segment_prob))
st.caption(f"Confidence score for the selected customer: {segment_prob:.1%}")

badge_fg, badge_bg = segment_badge(cluster_name)
render_badge(cluster_name, badge_fg, badge_bg)

st.write("")
meta_cols = st.columns(5)
meta_cols[0].metric("Customer ID", str(selected_id))
meta_cols[1].metric("Recency", f"{int(customer_row['Recency'])} days")
meta_cols[2].metric("Frequency", f"{int(customer_row['Frequency'])}")
meta_cols[3].metric("Avg Monthly Spend", f"USD {customer_row['Avg_Monthly_Spend']:.0f}")
meta_cols[4].metric("Returns", f"{int(customer_row['Return_Count'])}")

st.divider()

# --- INSIGHT & ACTION ---
roi_status, roi_desc = calculate_roi_strategy(segment_prob, real_spend, cluster_name)
drift = customer_row['Velocity_Drift']

# Insight Logic
if cluster_name == "High Value" and drift < -0.15:
    insight_msg = "FALLING VIP"
    style = "error"
elif cluster_name == "At-Risk" and drift > 0.15:
    insight_msg = "RECOVERING CUSTOMER"
    style = "success"
elif customer_row["Recency"] > 180:
    insight_msg = "LONG INACTIVE"
    style = "error"
elif drift <= -0.15:
    insight_msg = "COOLING DOWN"
    style = "warning"
elif drift >= 0.15:
    insight_msg = "GAINING MOMENTUM"
    style = "success"
elif customer_row["Frequency"] <= 2:
    insight_msg = "NEW / INFREQUENT CUSTOMER"
    style = "info"
elif value_pct >= 0.90:
    insight_msg = "LOYAL ANCHOR"
    style = "success"
else:
    insight_msg = "STABLE BEHAVIOR"
    style = "info"

c_insight, c_roi = st.columns([2, 1])
with c_insight:
    getattr(st, style)(f"**Insight**: {insight_msg}")
with c_roi:
    roi_fg, roi_bg = action_badge(roi_status)
    st.metric("Recommended Action", roi_status, help=roi_desc)
    render_badge(roi_status, roi_fg, roi_bg)

# --- SHAP DRIVER ANALYSIS ---
st.subheader("Driver Analysis")

def render_shap_plot(shap_values, title, positive_color, negative_color):
    """Helper to render standardized SHAP bar charts."""
    fig, ax = plt.subplots(figsize=(5,3))
    
    shap_flat = np.array(shap_values).flatten()
    if len(shap_flat) > len(FEATURES): shap_flat = shap_flat[:len(FEATURES)]
    
    indices = np.argsort(np.abs(shap_flat))[-5:] 
    plot_vals = shap_flat[indices]
    plot_names = np.array(FEATURES)[indices]
    
    colors = [positive_color if x > 0 else negative_color for x in plot_vals]
    plt.barh(plot_names, plot_vals, color=colors)
    plt.title(title)
    plt.xlabel("Impact Magnitude")
    plt.tight_layout()
    
  
    clean_data = [(name, f"{val:.2f}") for name, val in zip(plot_names, plot_vals)]
    
    return fig, clean_data

c_risk, c_spend = st.columns(2)

# Risk Drivers
shap_risk_raw = explainer_clf.shap_values(X.iloc[[customer_idx]])
if isinstance(shap_risk_raw, list): shap_risk_raw = shap_risk_raw[pred_cluster]

with c_risk:
    fig_r, risk_drivers = render_shap_plot(shap_risk_raw, f"Why {cluster_name}?", "#D9534F", "#6B7280")
    st.pyplot(fig_r)

# Spend Drivers
shap_spend_raw = explainer_reg.shap_values(X.iloc[[customer_idx]])
with c_spend:
    fig_s, spend_drivers = render_shap_plot(shap_spend_raw, "Spend Factors", "#2E8B57", "#6B7280")
    st.pyplot(fig_s)

# --- STRATEGY GENERATION ---
st.divider()
st.subheader("Executive Strategy")

if st.button("Generate Executive Report", use_container_width=True):
    with st.spinner("Analyzing behavioral patterns..."):
        context = {
            "id": selected_id,
            "segment": cluster_name,
            "prob": f"{segment_prob:.1%}",
            "spend": f"${real_spend:.0f}",
            "percentile": tier,
            "insight": insight_msg,
            "roi_action": f"{roi_status} {roi_desc}",
            "risk_drivers": str(risk_drivers),
            "spend_drivers": str(spend_drivers)
        }
        
        raw_advice = generate_strategic_narrative(context)
        
        # Sanitize output (prevents Markdown rendering issues with $)
        clean_advice = raw_advice.replace("$", " USD ")
        render_strategy_output(clean_advice)

    # Technical details panel
    with st.expander("Technical Details"):
        st.write("Feature Vector:")
        st.dataframe(customer_row.to_frame().T)

    st.markdown(
        '<div class="footer-note">Powered by XGBoost • SHAP • Groq Llama 3 • Streamlit</div>',
        unsafe_allow_html=True,
    )