import pickle
import os
import re
from typing import Dict, Any
from collections import defaultdict
import pandas as pd
import numpy as np

# --- State Management & Artifact Loading ---
_cache: Dict[str, Any] = {}
ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'assets')

def _load_artifact(filename: str) -> Any:
    """Loads a .pkl artifact from the assets directory, with caching."""
    if filename in _cache:
        return _cache[filename]
    
    path = os.path.join(ASSETS_DIR, filename)
    if not os.path.exists(path):
        # A real CLI would use a custom exception and rich-traceback
        print(f"FATAL ERROR: Artifact {filename} not found in {ASSETS_DIR}")
        raise FileNotFoundError(f"Missing required asset: {filename}")
    
    try:
        with open(path, 'rb') as f:
            artifact = pickle.load(f)
            _cache[filename] = artifact
            print(f"Successfully loaded and cached '{filename}'")
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

def get_historical_report(ticker: str, horizon_str: str) -> pd.DataFrame:
    """
    Loads and returns the pre-computed historical certainty report.
    """
    historical_reports = _load_artifact('historical_certainty_reports.pkl')
    
    if ticker not in historical_reports or horizon_str not in historical_reports[ticker]:
        return pd.DataFrame()
        
    return historical_reports[ticker][horizon_str]