import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os
from groq import Groq

# CONFIGURATION & AUTHENTICATION
st.set_page_config(page_title="Behavioural Intelligence Engine", layout="wide", page_icon=r"C:\Users\arnav\OneDrive\Documents\3dc3f966-9b4e-4f1a-9ef9-02ee0a3f62b3.jpeg")

# API Key Validation
if "GROQ_API_KEY" in st.secrets:
    API_KEY = st.secrets["GROQ_API_KEY"]
else:
    st.error("Missing API Key. Please configure .streamlit/secrets.toml")
    st.stop()

client = Groq(api_key=API_KEY)

# Path Configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'processed', 'labeled_customers.csv')
MODEL_CLASS_PATH = os.path.join(BASE_DIR, '..', 'models', 'xgb_classifier.json')
MODEL_REG_PATH = os.path.join(BASE_DIR, '..', 'models', 'xgb_regressor.json')

# RESOURCE MANAGEMENT (CACHED)
@st.cache_data
def load_data():
    """Loads and caches the customer dataset."""
    if not os.path.exists(DATA_PATH):
        st.error(f"Data file not found at: {DATA_PATH}")
        st.stop()
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    """Loads trained XGBoost models from disk."""
    if not os.path.exists(MODEL_CLASS_PATH) or not os.path.exists(MODEL_REG_PATH):
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
    GROSS_MARGIN = 0.30
    potential_profit = potential_spend * GROSS_MARGIN
    
    if segment_name == 'At-Risk':
        cost = 15.0
        captured_value = potential_profit * 0.30
        strategy = "DEFEND"
    elif segment_name == 'High Value':
        cost = 3.0
        captured_value = potential_profit * 0.10
        strategy = "UPSELL"
    else:
        cost = 5.0
        captured_value = potential_profit * 0.15
        strategy = "NURTURE"

    roi = captured_value - cost
    
    if roi > 0:
        return f"{strategy}", f"Est. ROI: +${roi:.0f} (Cost: ${cost})"
    return "SKIP", f"Value (${captured_value:.0f}) < Cost (${cost})"

def generate_strategic_narrative(context_data):
    """Uses LLM to generate a qualitative strategic report."""
    prompt = f"""
    Role: Senior Retention Strategist.
    
    Client Profile:
    - ID: {context_data['id']}
    - Segment: {context_data['segment']} (Conf: {context_data['prob']})
    - Future Value: {context_data['spend']} ({context_data['percentile']})
    - Momentum: {context_data['insight']}
    - Action: {context_data['roi_action']}
    
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
col1.metric("Segment", cluster_name)
col2.metric("Confidence", f"{segment_prob:.1%}")
col3.metric("Predicted Spend (90d)", f"${real_spend:.0f}")

if value_pct >= 0.9: 
    tier, color = "Top 10%", "normal"
elif value_pct >= 0.7: 
    tier, color = "Top 30%", "normal"
elif value_pct >= 0.5: 
    tier, color = "Mid Tier", "off"
else: 
    tier, color = "Low Priority", "off"
col4.metric("Value Tier", tier, f"{value_pct:.0%} Percentile", delta_color=color)

st.divider()

# --- INSIGHT & ACTION ---
roi_status, roi_desc = calculate_roi_strategy(segment_prob, real_spend, cluster_name)
drift = customer_row['Velocity_Drift']

# Insight Logic
if cluster_name == 'High Value' and (real_spend < 200 or drift < -0.1):
    insight_msg = "FALLING VIP (Slowing Down)"
    style = "error"
elif cluster_name == 'At-Risk' and (real_spend > 300 or drift > 0.1):
    insight_msg = "RISING STAR (Reactivation Detected)"
    style = "success"
elif drift > 0.5:
    insight_msg = "GAINING MOMENTUM (Velocity ↑)"
    style = "success"
elif drift < -0.5:
    insight_msg = "COOLING DOWN (Velocity ↓)"
    style = "warning"
elif value_pct > 0.9:
    insight_msg = "LOYAL ANCHOR (Consistent VIP)"
    style = "success"
else:
    insight_msg = "STABLE BEHAVIOR"
    style = "info"

c_insight, c_roi = st.columns([2, 1])
with c_insight:
    getattr(st, style)(f"**Insight**: {insight_msg}")
with c_roi:
    st.metric("Recommended Action", roi_status, help=roi_desc)

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
    fig_r, risk_drivers = render_shap_plot(shap_risk_raw, f"Why {cluster_name}?", "red", "blue")
    st.pyplot(fig_r)

# Spend Drivers
shap_spend_raw = explainer_reg.shap_values(X.iloc[[customer_idx]])
with c_spend:
    fig_s, spend_drivers = render_shap_plot(shap_spend_raw, "Spend Factors", "green", "gray")
    st.pyplot(fig_s)

# --- STRATEGY GENERATION ---
st.divider()
st.subheader("Executive Strategy")

if st.button("Generate Strategy"):
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
        st.markdown(clean_advice)

# Debug Panel
with st.expander("System Internals (Debug View)"):
    st.write("Feature Vector:")
    st.dataframe(customer_row.to_frame().T)