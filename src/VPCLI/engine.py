import pickle
import os
import re
import warnings
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import pandas as pd
import numpy as np
from .core import constants

# --- State Management & Artifact Loading ---
_cache: Dict[str, Any] = {}
ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'assets')

def _load_artifact(filename: str) -> Any:
    """Loads a .pkl artifact from the assets directory, with caching."""
    if filename in _cache:
        return _cache[filename]
    
    path = os.path.join(ASSETS_DIR, filename)
    if not os.path.exists(path):
        print(f"FATAL ERROR: Artifact {filename} not found in {ASSETS_DIR}")
        raise FileNotFoundError(f"Missing required asset: {filename}")
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            with open(path, 'rb') as f:
                artifact = pickle.load(f)
        
        _cache[filename] = artifact
        return artifact
    except Exception as e:
        print(f"ERROR loading {filename}: {e}")
        raise

# --- Logic Transplanted from Notebook ---
def simulate_investment_path(ticker_pipelines, modeling_datasets_ticker, path_horizons, top_n_features=5, show_volatility=False):
    """
    Simulates investment rules by generating scenarios and predicting a performance path.
    (This is the exact code from notebook cell 6.7.0)
    """
    if not ticker_pipelines:
        return pd.DataFrame()
    agg_importances = defaultdict(float)
    for horizon in path_horizons:
        if horizon in ticker_pipelines and ticker_pipelines[horizon]['models'].get('model_q50'):
            model = ticker_pipelines[horizon]['models']['model_q50']
            lagged_feature_names = modeling_datasets_ticker[horizon]['X_train'].columns
            importances = model.feature_importances_
            for lagged_name, importance_value in zip(lagged_feature_names, importances):
                base_name = lagged_name.split('_lag')[0]
                agg_importances[base_name] += importance_value
    if not agg_importances:
        return pd.DataFrame()
    top_base_features = sorted(agg_importances, key=agg_importances.get, reverse=True)[:top_n_features]
    scenarios = []
    for base_feature in top_base_features:
        target_horizon = path_horizons[-1]
        target_lagged_col = [col for col in modeling_datasets_ticker[target_horizon]['X_train'].columns if base_feature in col]
        if not target_lagged_col: continue
        
        lag_match = re.search(r'_lag(-?\d+)$', target_lagged_col[0])
        optimal_lag = int(lag_match.group(1)) if lag_match else 'N/A'
        ref_df = modeling_datasets_ticker[target_horizon]['X_train']
        quantiles = np.linspace(0, 1, 11)
        scenario_values = ref_df[target_lagged_col[0]].quantile(quantiles)
        for i in range(10):
            low_bound, high_bound = scenario_values.iloc[i], scenario_values.iloc[i+1]
            scenario_val = (low_bound + high_bound) / 2
            
            scenario_result = {
                'Signal_Feature': base_feature,
                'Optimal_Lag': f"{optimal_lag}d",
                'Signal_Interval': (low_bound, high_bound)
            }
            for horizon in path_horizons:
                if horizon not in ticker_pipelines: continue
                
                models = ticker_pipelines[horizon]['models']
                X_train_ref = modeling_datasets_ticker[horizon]['X_train']
                baseline_vector = X_train_ref.median().to_dict()
                
                current_lagged_col = [col for col in X_train_ref.columns if base_feature in col]
                if not current_lagged_col: continue
                baseline_vector[current_lagged_col[0]] = scenario_val
                scenario_vector_df = pd.DataFrame([baseline_vector], columns=X_train_ref.columns)
                
                if models.get('model_q50'):
                    scenario_result[f'Est_ROI_{horizon}'] = models['model_q50'].predict(scenario_vector_df)[0]
                if models.get('model_prob'):
                    scenario_result[f'Prob_Gain_{horizon}'] = models['model_prob'].predict_proba(scenario_vector_df)[0, 1]
                
                if show_volatility and models.get('model_vol'):
                    scenario_result[f'Est_Volatility_{horizon}'] = models['model_vol'].predict(scenario_vector_df)[0]
            scenarios.append(scenario_result)
            
    return pd.DataFrame(scenarios)

# --- Public Engine API ---
def get_predictive_report(ticker: str, horizon_str: str, show_volatility: bool = False) -> pd.DataFrame:
    """
    Orchestrates the loading of predictive artifacts and runs the simulation.
    """
    trained_pipelines = _load_artifact('trained_pipelines.pkl')
    modeling_datasets = _load_artifact('modeling_datasets.pkl')

    if ticker not in trained_pipelines or ticker not in modeling_datasets:
        return pd.DataFrame()
        
    horizon_val = int(horizon_str.replace('M', ''))
    ticker_pipelines = trained_pipelines[ticker]
    path_horizons = sorted([h for h in ticker_pipelines.keys() if int(h[:-1]) <= horizon_val], key=lambda x: int(x[:-1]))
    
    rules_df = simulate_investment_path(
        ticker_pipelines, 
        modeling_datasets[ticker], 
        path_horizons, 
        show_volatility=show_volatility
    )
    # Attach path_horizons for the display logic to use
    rules_df.attrs['path_horizons'] = path_horizons
    return rules_df

# --- REFACTORED Historical Report Engine Logic ---
def get_monthly_signal_summary(
    ticker: str, horizon_str: str
) -> List[Tuple[pd.Timestamp, pd.DataFrame]]:
    """
    Loads, processes, and aggregates historical signal data by month.
    """
    # 1. Load all necessary artifacts
    historical_reports = _load_artifact("historical_certainty_reports.pkl")
    engine_dfs = _load_artifact("engine_dfs_engineered.pkl")

    if (
        ticker not in historical_reports
        or horizon_str not in historical_reports[ticker]
        or ticker not in engine_dfs
    ):
        return []

    # 2. Filter raw report for unique, high-certainty signals
    raw_report_df = historical_reports[ticker][horizon_str]
    raw_report_df = raw_report_df[
        ~raw_report_df["Signal_Feature"].astype(str).str.contains("SIGNALS ACTIVE IN:", na=False)
    ].copy()
    unique_signals_df = raw_report_df.groupby(
        ["Signal_Feature", "Signal_Interval"]
    ).size().reset_index(name="count")

    # 3. Map all historical occurrences of these signals
    all_occurrences = []
    company_df = engine_dfs[ticker].copy()
    company_df.reset_index(inplace=True)
    company_df['Period'] = pd.to_datetime(company_df['Period'])
    max_date = company_df['Period'].max()

    for _, signal in unique_signals_df.iterrows():
        feature = signal["Signal_Feature"]
        interval = signal["Signal_Interval"]
        occurrences = company_df[
            company_df[feature].between(interval[0], interval[1], inclusive="both")
        ].copy()
        
        # CORRECTED: Assign values in a way that pandas handles correctly by matching the index length.
        occurrences["Signal_Feature"] = [feature] * len(occurrences)
        occurrences["Signal_Interval"] = [interval] * len(occurrences)
        all_occurrences.append(occurrences)

    if not all_occurrences:
        return []

    master_df = pd.concat(all_occurrences, ignore_index=True)

    # 4. Defensive Horizon Cutoff: Invalidate impossible future ROIs
    horizon_months = int(horizon_str.replace("M", ""))
    target_return_col = f"Future_Return_{horizon_str}"
    
    # Vectorized approach for performance
    master_df['cutoff_date'] = master_df['Period'] + pd.DateOffset(months=horizon_months)
    master_df.loc[master_df['cutoff_date'] > max_date, target_return_col] = np.nan
    master_df.drop(columns=['cutoff_date'], inplace=True)


    # 5. Group by month and aggregate signals
    monthly_groups = master_df.groupby(pd.Grouper(key="Period", freq="ME"))
    
    monthly_summaries = []
    for month, month_df in monthly_groups:
        if month_df.empty:
            continue

        agg_funcs = {
            target_return_col: [
                "mean", "min", "max", "std", 
                lambda x: np.nansum(x > 0), 
                lambda x: np.nansum(x <= 0)
            ]
        }
        agg_df = month_df.groupby(["Signal_Feature", "Signal_Interval"]).agg(agg_funcs).reset_index()
        agg_df.columns = ["_".join(col).strip() for col in agg_df.columns.values]
        
        rename_map = {
            "Signal_Feature_": "Signal_Feature",
            "Signal_Interval_": "Signal_Interval",
            f"{target_return_col}_mean": f"Hist. Avg. ROI @ {horizon_str}",
            f"{target_return_col}_min": f"Min ROI @ {horizon_str}",
            f"{target_return_col}_max": f"Max ROI @ {horizon_str}",
            f"{target_return_col}_std": f"Std Dev ROI @ {horizon_str}",
            f"{target_return_col}_<lambda_0>": "Win_Count",
            f"{target_return_col}_<lambda_1>": "Loss_Count"
        }
        agg_df = agg_df.rename(columns=rename_map)
        
        agg_df["Occurrences"] = agg_df["Win_Count"] + agg_df["Loss_Count"]
        agg_df["Win_Rate"] = (agg_df["Win_Count"] / agg_df["Occurrences"]).fillna(0)
        agg_df["Win/Loss Ratio"] = [
            f"{wr:.1%} ({int(w)} W / {int(l)} L)"
            for wr, w, l in zip(agg_df["Win_Rate"], agg_df["Win_Count"], agg_df["Loss_Count"])
        ]
        
        # Only add the monthly summary if it contains actual data
        if agg_df["Occurrences"].sum() > 0:
            final_month_df = agg_df.sort_values(by=["Win_Rate", "Occurrences"], ascending=[False, False])
            monthly_summaries.append((month, final_month_df))

    return sorted(monthly_summaries, key=lambda x: x[0], reverse=True)