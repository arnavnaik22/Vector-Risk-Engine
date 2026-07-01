# Vector | Hybrid Behavioral Intelligence Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![Llama 3](https://img.shields.io/badge/GenAI-Llama%203-purple)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Research%20Artifact-orange)

Vector is a hybrid behavioral intelligence engine for customer risk and value forecasting. It combines engineered behavioral signals, XGBoost-based inference, SHAP explanations, and a Groq-hosted Llama 3 assistant to generate human-readable retention actions.

The repository is intended for human-in-the-loop decision support, not autonomous intervention.

## What Vector Does

Vector goes beyond static churn scores by tracking movement in customer behavior over time. It uses velocity-style features to identify customers who are cooling down, stabilizing, or gaining momentum, then pairs those signals with a lightweight business rule layer and a generative strategy layer.

## Repository Contents

- `app/`: Streamlit dashboard for interactive customer analysis.
- `data/raw/`: UCI Online Retail source data used by the pipeline.
- `data/processed/`: Prebuilt customer feature table used by the app.
- `models/`: Trained XGBoost classifier and regressor artifacts.
- `experiments/`: Offline research pipeline, ablation study, and exported charts.
- `notebooks/`: Segmentation, churn prediction, and SHAP explanation notebooks.
- `paper.pdf`: Research paper describing the framework and evaluation.
- `assets/`: UI assets used by the dashboard.
- `runtime.txt`: Python version for Streamlit Cloud.

## Core Pipeline

### 1. Feature Engineering

- Uses a strict 90-day split to reduce leakage.
- Builds RFM-style features plus velocity signals such as `Velocity_Recent` and `Velocity_Drift`.
- Derives `Avg_Monthly_Spend` and `AOV` so value and tenure are modeled separately.

### 2. Inference

- An XGBoost classifier assigns a behavioral segment.
- An XGBoost regressor predicts future spend.
- SHAP explainers expose the strongest drivers behind each prediction.

### 3. Strategy Generation

- A deterministic ROI gate filters out actions that do not clear the cost threshold.
- A Groq Llama 3 model turns the numeric signals into concise tactical guidance.

## Installation

The project targets Python 3.11 for Streamlit Cloud.

Install the runtime packages with:

```bash
pip install streamlit pandas numpy xgboost shap matplotlib groq scikit-learn seaborn openpyxl
```

## Required Artifacts

The dashboard loads these committed artifacts directly:

- `data/raw/Online_Retail.xlsx`
- `data/processed/labeled_customers.csv`
- `models/xgb_classifier.json`
- `models/xgb_regressor.json`

If you retrain the pipeline, regenerate the processed CSV and model files before pushing.

## Configure Secrets

Create `.streamlit/secrets.toml` with your Groq key for the optional narrative generator:

```toml
GROQ_API_KEY = "your_key_here"
```

## Run the Dashboard

```bash
streamlit run app/main.py
```

## Streamlit Cloud

Use `app/main.py` as the entrypoint and add `GROQ_API_KEY` in the Streamlit Cloud secrets UI. If the key is missing, the dashboard still loads and uses a local fallback for strategy text.

## Run The Research Pipeline

The offline experiment script reproduces the feature engineering and model comparison workflow:

```bash
python experiments/04_research.py
```

It writes summary tables and charts into `experiments/`.

## Example Outputs

- `experiments/final_research_results.csv`
- `experiments/final_chart_clean.png`
- `experiments/ablation_summary.csv`
- `notebooks/shap_sales_summary.png`
- `notebooks/shap_risk_summary.png`

## Limitations

- The model estimates behavioral risk, not causal treatment effect.
- Drift thresholds are tuned for the UCI Online Retail data distribution and may need recalibration for other domains.
- Real-time SHAP analysis adds latency, so precomputation may be preferable for higher-throughput use cases.

## Citation

If you use Vector as a research artifact, reference the paper included in this repository and describe any changes you made to the feature engineering or training setup.
