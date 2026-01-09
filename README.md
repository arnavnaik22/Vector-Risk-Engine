# Vector | Hybrid Behavioral Intelligence Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![Llama 3](https://img.shields.io/badge/GenAI-Llama%203-purple)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Research%20Artifact-orange)

> **A production-oriented decision support system bridging the gap between quantitative risk prediction and qualitative strategic reasoning.**
>
> *Vector is designed for human-in-the-loop decision support, not autonomous intervention.*

**Vector** moves beyond static churn probability. By synthesizing **differential behavioral vectors** (Velocity Drift) with **economic constraints** and **Generative AI**, it identifies *why* a customer is changing trajectory and *automatically* drafts the optimal retention strategy.

---

### Research Context
This repository serves as the official implementation artifact for the research report:
**"Behavioral Intelligence Engine: A Hybrid Predictive–Generative Framework for CLV"**

The paper details the mathematical foundations of Entropic Stability, the ablation studies on Velocity Drift, and the hybrid predictive–generative architecture used in this codebase.

**[Download the Full Research Paper (DOCX)](paper.pdf)**

---

## The Core Problem
Standard RFM (Recency, Frequency, Monetary) models are **static snapshots**. They cannot distinguish between a customer who is *accumulating* value and one who is *decelerating*.
* **The Gap:** High-value customers often show "negative momentum" (drift) long before they churn.
* **The Solution:** Vector treats customer behavior as a physics problem—measuring the **velocity** and **acceleration** of spend—gated by a rigorous economic ROI filter.

## System Architecture

Vector operates as a decoupled, three-stage pipeline designed for interpretability and latency-constrained environments.

### 1. The Refinery (Feature Engineering)
* **Temporal Causality:** Enforces a strict 90-day lookahead window to prevent data leakage.
* **Vector Drift ($\Delta v$):** Computes the differential between *Recent Velocity* (90-day) and *Lifetime Velocity*. This acts as a leading indicator for "Silent Attrition."
* **Decoupled Value:** Replaces lifetime aggregates with `Avg_Monthly_Spend` to separate tenure from intensity.

### 2. The Engine (Inference Core)
* **Behavioral Regimes:** K-Means clustering on log-normalized manifolds to identify stable behavioral archetypes.
* **Risk Classification:** An XGBoost Classifier approximates cluster boundaries (96% Accuracy) for real-time segmentation.
* **Value Forecasting:** An XGBoost Regressor optimized for **Ranking Quality** (Spearman Rho) rather than raw error, prioritizing the *relative* order of high-value customers.

### 3. The Strategist (Generative Decision Layer)
* **Economic Gating:** A deterministic logic layer that suppresses interventions where `Cost > Predicted_LTV * Margin`.
* **Generative Reasoning:** A **Llama-3 (Groq)** agent acts as a "Senior Analyst." It ingests the numeric risk signals + SHAP drivers to generate structured, tactical recommendations (e.g., "Defend," "Nurture," "Upsell").

---

## Performance Benchmarks

Retail data is notoriously heavy-tailed ("Whales" distort metrics). Vector is evaluated on **Operational Utility** (Ranking) rather than just statistical fit.

| Metric | Score | Industrial Context |
| :--- | :--- | :--- |
| **Ranking Quality** | **0.51–0.56 (Spearman)** | Strong signal for prioritizing high-value lists. |
| **Risk Precision** | **98% (At-Risk)** | Near-zero waste on false-positive retention offers. |
| **Error Reduction** | **29.4%** | Improvement over baseline mean-value predictors. |
| **Core RMSE** | **$785** | Accuracy on the "Core Business" (99% of users). |

*> **Note:** Raw RMSE ($7,540) is reported in the paper for transparency but excluded from optimization, as it is driven by top 1% outliers.*

---

## Key Capabilities

### Momentum Detection
Static models miss gradual disengagement. Vector monitors the first derivative of purchase frequency:
* `Drift < -0.5`: **"Cooling Down"** (Immediate intervention candidate).
* `Drift > +0.5`: **"Gaining Momentum"** (Prime upsell candidate).

### ROI-Gated Logic
The system is economically aware. It will **suppress** high-risk alerts if the math doesn't work:
```python
if (Predicted_LTV * Margin * Capture_Rate) > Intervention_Cost:
    return "Actionable"
else:
    return "Suppress (Negative ROI)"
```

Here is the clean, professional version of the README with all emojis removed, ready for your repository.

README.md
Markdown

# Vector | Hybrid Behavioral Intelligence Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![Llama 3](https://img.shields.io/badge/GenAI-Llama%203-purple)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Research%20Artifact-orange)

> **A production-oriented decision support system bridging the gap between quantitative risk prediction and qualitative strategic reasoning.**
>
> *Vector is designed for human-in-the-loop decision support, not autonomous intervention.*

**Vector** moves beyond static churn probability. By synthesizing **differential behavioral vectors** (Velocity Drift) with **economic constraints** and **Generative AI**, it identifies *why* a customer is changing trajectory and *automatically* drafts the optimal retention strategy.

---

### Research Context
This repository serves as the official implementation artifact for the research report:
**"Behavioral Intelligence Engine: A Hybrid Predictive–Generative Framework for CLV"**

The paper details the mathematical foundations of Entropic Stability, the ablation studies on Velocity Drift, and the hybrid predictive–generative architecture used in this codebase.

**[Download the Full Research Paper (DOCX)](paper.docx)**

---

## The Core Problem
Standard RFM (Recency, Frequency, Monetary) models are **static snapshots**. They cannot distinguish between a customer who is *accumulating* value and one who is *decelerating*.
* **The Gap:** High-value customers often show "negative momentum" (drift) long before they churn.
* **The Solution:** Vector treats customer behavior as a physics problem—measuring the **velocity** and **acceleration** of spend—gated by a rigorous economic ROI filter.

## System Architecture

Vector operates as a decoupled, three-stage pipeline designed for interpretability and latency-constrained environments.

### 1. The Refinery (Feature Engineering)
* **Temporal Causality:** Enforces a strict 90-day lookahead window to prevent data leakage.
* **Vector Drift ($\Delta v$):** Computes the differential between *Recent Velocity* (90-day) and *Lifetime Velocity*. This acts as a leading indicator for "Silent Attrition."
* **Decoupled Value:** Replaces lifetime aggregates with `Avg_Monthly_Spend` to separate tenure from intensity.

### 2. The Engine (Inference Core)
* **Behavioral Regimes:** K-Means clustering on log-normalized manifolds to identify stable behavioral archetypes.
* **Risk Classification:** An XGBoost Classifier approximates cluster boundaries (96% Accuracy) for real-time segmentation.
* **Value Forecasting:** An XGBoost Regressor optimized for **Ranking Quality** (Spearman Rho) rather than raw error, prioritizing the *relative* order of high-value customers.

### 3. The Strategist (Generative Decision Layer)
* **Economic Gating:** A deterministic logic layer that suppresses interventions where `Cost > Predicted_LTV * Margin`.
* **Generative Reasoning:** A **Llama-3 (Groq)** agent acts as a "Senior Analyst." It ingests the numeric risk signals + SHAP drivers to generate structured, tactical recommendations (e.g., "Defend," "Nurture," "Upsell").

---

## Performance Benchmarks

Retail data is notoriously heavy-tailed ("Whales" distort metrics). Vector is evaluated on **Operational Utility** (Ranking) rather than just statistical fit.

| Metric | Score | Industrial Context |
| :--- | :--- | :--- |
| **Ranking Quality** | **0.51–0.56 (Spearman)** | Strong signal for prioritizing high-value lists. |
| **Risk Precision** | **98% (At-Risk)** | Near-zero waste on false-positive retention offers. |
| **Error Reduction** | **29.4%** | Improvement over baseline mean-value predictors. |
| **Core RMSE** | **$785** | Accuracy on the "Core Business" (99% of users). |

*> **Note:** Raw RMSE ($7,540) is reported in the paper for transparency but excluded from optimization, as it is driven by top 1% outliers.*

---

## Key Capabilities

### Momentum Detection
Static models miss gradual disengagement. Vector monitors the first derivative of purchase frequency:
* `Drift < -0.5`: **"Cooling Down"** (Immediate intervention candidate).
* `Drift > +0.5`: **"Gaining Momentum"** (Prime upsell candidate).

### ROI-Gated Logic
The system is economically aware. It will **suppress** high-risk alerts if the math doesn't work:
```python
if (Predicted_LTV * Margin * Capture_Rate) > Intervention_Cost:
    return "Actionable"
else:
    return "Suppress (Negative ROI)"
```

First-Class Explainability
SHAP (SHapley Additive exPlanations) values are not an afterthought. They are computed per-instance to resolve conflicting signals (e.g., "Why is this VIP flagged as At-Risk?") and fed directly into the LLM context.

Installation & Usage
Prerequisites: Python 3.9+

Clone the Repository
git clone [https://github.com/YOUR_USERNAME/vector-risk-engine.git](https://github.com/YOUR_USERNAME/vector-risk-engine.git)
cd vector-risk-engine

Install Dependencies
pip install -r requirements.txt --upgrade

Configure Environment Create a .streamlit/secrets.toml file for the LLM inference layer:
GROQ_API_KEY = "your_key_here"

Launch the Dashboard
streamlit run app/main.py

Limitations & Scope
Observational Nature: The model estimates behavioral risk, not the causal impact of interventions (A/B testing required for validation).

Domain Calibration: Drift thresholds are calibrated on the UCI Online Retail dataset. New domains require re-tuning of the velocity parameters.

Inference Cost: Real-time SHAP calculation adds latency; for high-throughput production, pre-computation is recommended.

Repository Structure

notebooks/: Research sandbox (Segmentation, Ablation Studies, SHAP Analysis).

app/: Production-grade Streamlit dashboard.

models/: Serialized XGBoost artifacts (JSON).

data/: ETL scripts and schema definitions.

paper.docx: Full research documentation.

