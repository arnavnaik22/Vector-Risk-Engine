
# Vector: Behavioural Risk Engine

**Hybrid Decision Support System for Customer Retention & Near-Term LTV**

## Technical Summary

Vector is a production-oriented decision engine designed to prioritize customer retention actions under realistic data constraints. It identifies behaviorally at-risk customers and forecasts 90-day future spend while explicitly modeling momentum, explainability, and economic feasibility.

Unlike standard churn classifiers that output a static probability, Vector captures directional behaviour (velocity and drift), explains *why* risk emerges, and gates interventions through an ROI filter before recommending action.

*The system is observational, not causal, and is intended for decision support rather than automated intervention.*

---

## System Architecture

The pipeline is composed of three decoupled stages:

### 1. Refinery (Feature Engineering)

* **Temporal Splitting:** A strict 90-day lookahead window enforces causal validity and prevents future leakage.
* **Leakage Prevention:** Lifetime monetary aggregates are replaced with `Avg_Monthly_Spend` and `AOV` to decouple tenure from value.
* **Momentum Signals:** `Velocity_Drift` is computed as the difference between recent and lifetime purchase frequency to detect behavioral inflection points before spend collapses.

### 2. Engine (Hybrid Inference Core)

* **Behavioral Segmentation:** K-Means clustering on log-normalized RFM features discovers stable behavioural regimes. Chosen for interpretability and suitability for low-dimensional behavioral manifolds.
* **Risk Classification:** An XGBoost classifier approximates cluster boundaries for fast, real-time inference. *Note: The classifier learns segment structure, not ground-truth churn labels.*
* **Spend Forecasting:** An XGBoost regressor estimates 90-day future spend using a log-transformed target. Optimization prioritizes ranking quality over variance explanation.

### 3. Decision Layer (Economics & Strategy)

* **Economic Filtering:** Predictions are passed through a margin/cost model to suppress negative-ROI interventions.
* **Explainability as Output:** SHAP values are computed per instance to resolve conflicting signals (e.g., high predicted value with emerging risk drivers).
* **Strategy Generation:** Llama-3 (via Groq) generates tactical recommendations conditioned on segment confidence, SHAP drivers, predicted value, and ROI feasibility. Outputs are constrained and grounded, not free-form text.

---

## Performance & Evaluation

Retail spend data is zero-inflated and heavy-tailed. Variance-based metrics such as R-Squared and raw RMSE are dominated by extreme outliers ("whales") and are poor indicators of operational usefulness.

Vector uses a robust evaluation protocol focused on prioritization quality and economic error:

| Metric | Score | Context |
| --- | --- | --- |
| **Spearman Rank Correlation** | `0.51` | Strong signal for relative value prioritization. |
| **Classification Accuracy** | `96.1%` | High fidelity mapping to behavioral risk buckets. |
| **Baseline MAE** | `$1,287` | Mean-based predictor. |
| **Vector MAE** | `$909` | **29.4% error reduction** vs baseline. |
| **Capped RMSE (99%)** | `$785` | Accuracy over the "Core Business" (99% of users). |

*Note: Raw RMSE ($7,540) is reported for transparency but excluded from optimization, as it is driven by the top 1% of spenders.*

---

## Key Capabilities

### Momentum Detection

Static RFM analysis fails to detect gradual disengagement. Vector monitors `Velocity_Drift` to capture directional change:

* **Drift < -0.5:** "Cooling Down" (Intervention candidate)
* **Drift > +0.5:** "Gaining Momentum" (Upsell candidate)

### ROI-Gated Logic

The system does not recommend action solely based on risk.

* *If (Predicted_LTV * Margin * Capture_Rate) > Intervention_Cost* -> **Recommend Action**
* *Else* -> **Suppress (Do Not Disturb)**

This prevents over-intervention and protects high-value customers from unnecessary contact.

### Explainability

SHAP values are computed per instance to attribute risk drivers, identify conflicting behavioral signals, and support analyst-level inspection. Explainability is treated as a first-class output, not a post-hoc visualization.

---

## Known Limitations

1. **Observational, Not Causal:** The model estimates behavioural risk, not the causal impact of interventions.
2. **Dataset Dependency:** Drift thresholds and ROI parameters are calibrated on the UCI Online Retail dataset and require retuning for new domains.
3. **SHAP Cost:** Local SHAP explanations are computationally expensive; production deployment would require pre-computation or further TreeExplainer optimization.

---

## Installation & Usage

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/vector-risk-engine.git
cd vector-risk-engine

```

**2. Install dependencies**

```bash
pip install -r requirements.txt

```

**3. Configure API (Optional)**
Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_key_here"

```

**4. Run the Dashboard**

```bash
streamlit run app/main.py

```
Note on File Paths: The notebooks and app currently use absolute paths specific to the development environment. Before running, please update the BASE_DIR or file path variables in app/main.py and the notebooks to match your local directory structure.

---

## Repository Structure

* `notebooks/`: Segmentation logic, model training pipelines, and SHAP analysis.
* `app/`: Production Streamlit application source code.
* `models/`: Serialized XGBoost artifacts.
* `data/`: Processing scripts and schema definitions.

---

*Built for the purpose of operationalizing customer intelligence.*