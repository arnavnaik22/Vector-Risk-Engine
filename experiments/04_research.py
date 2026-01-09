import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr, entropy
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION & REPRODUCIBILITY
# ==========================================
# Set global seed for exact reproducibility
np.random.seed(42)

# Paths (Adjust to match your environment)
RAW_DATA_PATH = r'data/raw/Online_Retail.xlsx' 
OUTPUT_DIR = 'experiments'

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# ==========================================
# 2. THE RESEARCH ENGINE (Feature Logic)
# ==========================================
def engineer_research_features(df_raw):
    """
    Implements the 'Vector' Research Framework:
    1. Velocity Drift (Differential Signal)
    2. Entropy & Burstiness (Chaos Theory)
    3. Cohort Relativity (Peer Benchmarking)
    """
    print("--- [Engine] Engineering Research Features... ---")
    
    # Setup
    df = df_raw.copy()
    if 'InvoiceDate' in df.columns:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Filter valid sales
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)].dropna(subset=['CustomerID'])
    
    # Strict 90-Day Split
    max_date = df['InvoiceDate'].max()
    cutoff_date = max_date - pd.Timedelta(days=90)
    
    df_past = df[df['InvoiceDate'] <= cutoff_date].copy()
    df_future = df[df['InvoiceDate'] > cutoff_date].copy()
    
    # BASE RFM
    snapshot_date = cutoff_date + pd.Timedelta(days=1)
    rfm = df_past.groupby('CustomerID').agg({
        'InvoiceDate': [lambda x: (snapshot_date - x.max()).days, lambda x: (snapshot_date - x.min()).days],
        'InvoiceNo': 'nunique',
        'Quantity': 'sum',
        'UnitPrice': lambda x: (x * df_past.loc[x.index, 'Quantity']).sum()
    })
    rfm.columns = ['Recency', 'Tenure', 'Frequency', 'Total_Items', 'Monetary_Sum']
    
    # RESEARCH FEATURE 1: VELOCITY DRIFT
    rfm['Tenure_Capped'] = rfm['Tenure'].clip(lower=30)
    recent_start = cutoff_date - pd.Timedelta(days=90)
    recent_stats = df_past[df_past['InvoiceDate'] > recent_start].groupby('CustomerID')['InvoiceNo'].nunique()
    
    rfm['Velocity_Recent'] = recent_stats.reindex(rfm.index).fillna(0) / 3.0
    rfm['Velocity_Life'] = rfm['Frequency'] / (rfm['Tenure_Capped'] / 30.0)
    rfm['Velocity_Drift'] = rfm['Velocity_Recent'] - rfm['Velocity_Life']
    
    # RESEARCH FEATURE 2: ENTROPY & BURSTINESS (Chaos)
    def get_chaos_metrics(dates):
        if len(dates) < 3: return pd.Series([0, 0], index=['Entropy', 'Burstiness'])
        intervals = dates.sort_values().diff().dt.days.dropna()
        if len(intervals) == 0 or intervals.sum() == 0: return pd.Series([0, 0], index=['Entropy', 'Burstiness'])
        
        # Entropy
        counts = intervals.value_counts(bins=5, normalize=True)
        ent = entropy(counts + 1e-10)
        
        # Burstiness
        mu, sigma = intervals.mean(), intervals.std()
        burst = (sigma - mu) / (sigma + mu + 1e-6)
        
        return pd.Series([ent, burst], index=['Entropy', 'Burstiness'])

    print("   > Calculating Chaos Metrics (Entropy)...")
    chaos = df_past.groupby('CustomerID')['InvoiceDate'].apply(get_chaos_metrics).unstack().fillna(0)
    rfm = rfm.join(chaos)
    
    # TARGET GENERATION
    future_spend = df_future.assign(Spend=df_future['Quantity'] * df_future['UnitPrice']) \
                            .groupby('CustomerID')['Spend'].sum() \
                            .rename('FutureSales_Label')
    
    # Final Merge
    final_df = rfm.merge(future_spend, on='CustomerID', how='left').fillna(0)
    final_df['Avg_Monthly_Spend'] = final_df['Monetary_Sum'] / (final_df['Tenure_Capped'] / 30)
    final_df['AOV'] = final_df['Monetary_Sum'] / final_df['Frequency']
    
    return final_df

# ==========================================
# 3. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # --- A. DATA LOADING ---
    print("\n[1/4] Loading Data...")
    
    # Fallback logic for paths
    possible_paths = [
        r'data/raw/Online_Retail.xlsx',
        r'../data/raw/Online_Retail.xlsx',
        r'C:\Users\arnav\OneDrive\Desktop\customer-risk-agent\data\raw\Online_Retail .xlsx',
        r'data/raw/Online_Retail.csv'
    ]
    
    df_raw = None
    for path in possible_paths:
        if os.path.exists(path):
            print(f"   > Found data at: {path}")
            if path.endswith('.xlsx'):
                df_raw = pd.read_excel(path)
            else:
                df_raw = pd.read_csv(path, encoding='ISO-8859-1')
            break
            
    if df_raw is None:
        print("[Error] Could not find 'Online_Retail' data file. Please check paths.")
        sys.exit(1)

    # --- B. FEATURE ENGINEERING ---
    print("\n[2/4] Generating Research Features...")
    df = engineer_research_features(df_raw)
    
    # --- C. ABLATION EXPERIMENT ---
    print("\n[3/4] Running Ablation Study (Comparison)...")
    
    # Config
    F_STATIC = ['Recency', 'Frequency', 'Avg_Monthly_Spend', 'Tenure', 'AOV']
    F_DRIFT  = F_STATIC + ['Velocity_Recent', 'Velocity_Drift']
    F_CHAOS  = F_DRIFT + ['Entropy', 'Burstiness'] # Full "Vector" Research Set

    y = np.log1p(df['FutureSales_Label'])

    models = {
        'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
        'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42),
        'Feedforward NN': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    print(f"\n{'Model':<15} | {'Set':<10} | {'MAE':<8} | {'Spearman':<8}")
    print("-" * 55)

    for m_name, model in models.items():
        for f_name, f_cols in [('Static', F_STATIC), ('+Drift', F_DRIFT), ('+Chaos', F_CHAOS)]:
            
            mae_scores = []
            spearman_scores = []
            
            X_raw = df[f_cols].values
            y_raw = y.values
            
            for train_idx, test_idx in kf.split(X_raw):
                X_train, X_test = X_raw[train_idx], X_raw[test_idx]
                y_train, y_test = y_raw[train_idx], y_raw[test_idx]
                
                # CRITICAL: Anti-Leakage Scaling
                if 'NN' in m_name:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                
                model.fit(X_train, y_train)
                
                # Prediction with Safety Clipping (Prevents Exploding Gradients)
                preds_log = model.predict(X_test)
                max_log_train = y_train.max() * 1.5 
                preds_log = np.clip(preds_log, 0, max_log_train)
                
                preds_real = np.expm1(preds_log)
                y_real = np.expm1(y_test)
                
                mae_scores.append(mean_absolute_error(y_real, preds_real))
                
                if np.std(preds_real) > 1e-9:
                    corr, _ = spearmanr(y_real, preds_real)
                    spearman_scores.append(corr)
                else:
                    spearman_scores.append(0.0)

            avg_mae = np.mean(mae_scores)
            avg_rho = np.mean(spearman_scores)
            
            results.append({
                'Model': m_name,
                'Features': f_name,
                'MAE': avg_mae,
                'Spearman': avg_rho
            })
            print(f"{m_name:<15} | {f_name:<10} | {avg_mae:.0f}     | {avg_rho:.3f}")

    # --- D. SAVE & PLOT ---
    print("\n[4/4] Saving Artifacts...")
    df_res = pd.DataFrame(results)
    df_res.to_csv(f'{OUTPUT_DIR}/final_research_results.csv', index=False)

    # Plot Spearman (Robust Metric)
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.barplot(data=df_res, x='Model', y='Spearman', hue='Features', palette='viridis')
    plt.ylim(0.45, 0.60) # Optimize scale to show differences
    plt.title('Impact of Higher-Order Features on Ranking Quality (Spearman)')
    plt.ylabel('Spearman Rho (Higher is Better)')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/final_chart_clean.png', dpi=300)
    
    print(f"--- SUCCESS: Chart saved to {OUTPUT_DIR}/final_chart_clean.png ---")