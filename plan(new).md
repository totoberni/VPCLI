Excellent. Now that the notebook proof-of-concept is complete and the environment is set, we can create a definitive, actionable implementation plan to migrate the engine's logic into the CLI tool.

I have analyzed the existing `plan.md`, the CPU-focused `plan(new).md`, and the provided notebook code. By merging these sources, I've created a single, cohesive, step-by-step plan that will take you from the current prototype to the final, functional application.

This plan prioritizes the **CPU-only architecture** for maximum portability, provides the exact code to transplant from your notebook, and translates the notebook's visualization logic into terminal-friendly output using `rich`.

Here is the new, consolidated implementation plan.

---

## Financial Strategist CLI: Final Implementation Plan

### **Phase 1: Finalize Project Environment (CPU-First)**

Our primary goal is portability. We must configure the environment to use a CPU-only version of `xgboost` and a generic Python Docker image, completely removing the NVIDIA CUDA dependency for deployment.

**Action 1.1: Update `pyproject.toml`**

We will enforce the CPU-only variant of `xgboost`. We will also keep `matplotlib` solely for its colormap utilities, which we'll use to style the `rich` tables.

*   **File:** `pyproject.toml`
*   **Action:** Replace the `dependencies` list with the following.

```toml:pyproject.toml
// ... existing code ...
dependencies = [
    # --- CLI Framework & Display ---
    "typer[all] >= 0.9.0",
    "rich >= 13.0.0",

    # --- Core Data & ML Libraries (Pinned for compatibility) ---
    "pandas == 2.3.3",
    "xgboost-cpu == 3.1.1",  # CRITICAL: Use the CPU-only package
    "scikit-learn == 1.7.2",
    "numpy == 2.3.4",
    
    # --- Styling Dependency ---
    "matplotlib == 3.10.7"  # Keep for color mapping logic
]
// ... existing code ...
```

**Action 1.2: Update `Dockerfile` for CPU Deployment**

We will replace the CUDA-based Docker image with a standard, lightweight Python image.

*   **File:** `Dockerfile`
*   **Action:** Replace the entire file content with the following.

```dockerfile:Dockerfile
# Stage 1: Use a standard, lightweight Python base image
FROM python:3.12-slim

# Stage 2: System Setup
# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off

# Set the working directory in the container
WORKDIR /app

# Stage 3: Application Setup
# Copy only the dependency definition first to leverage Docker's layer caching
COPY pyproject.toml .

# Install the project and its dependencies. This reads pyproject.toml.
RUN pip install .

# Copy the application source code and assets
COPY src/ ./src
COPY assets/ ./assets

# Stage 4: Execution
# Set the entrypoint to our installed script, making the container executable
ENTRYPOINT ["strategy-report"]

# Define the default command to run
CMD ["run"]
```

**Action 1.3: Rebuild the Local Environment**

After saving the changes above, run these commands in your terminal to apply them to your local `.venv`:

1.  `deactivate` (if active)
2.  `Remove-Item -Recurse -Force .venv, src\*.egg-info`
3.  `python -m venv .venv`
4.  `.\.venv\Scripts\Activate.ps1`
5.  `pip install -e .`

---

### **Phase 2: Implement the Core Engine (`src/VPCLI/engine.py`)**

This is where all the "thinking" happens. We will transplant the core simulation and data-loading logic from the notebook.

*   **File:** `src/VPCLI/engine.py`
*   **Action:** Replace the entire file content with the following code. This code includes a lazy-loading mechanism for artifacts, the `simulate_investment_path` function from your notebook, and the high-level API functions that will be called by `main.py`.

```python:src/VPCLI/engine.py
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
```

---

### **Phase 3: Implement Presentation Logic (`src/VPCLI/display.py`)**

Here we translate the `pandas.Styler` logic from the notebook into `rich.Table` objects for beautiful console output.

*   **File:** `src/VPCLI/display.py`
*   **Action:** Replace the entire file content with the following code. This provides the `rich`-native versions of both `display_historical_report` and `display_predictive_rulebook`.

```python:src/VPCLI/display.py
import pandas as pd
import re
import matplotlib
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

console = Console()

# --- Predictive Display Function (CLI Version) ---
def show_predictive_report(rules_df: pd.DataFrame, horizon_str: str, show_volatility: bool = False):
    if rules_df.empty:
        console.print("[yellow]No predictive rules could be generated for this scenario.[/yellow]")
        return
    
    # Determine Action
    rules_df['Action'] = 'Hold / Neutral'
    roi_col = f'Est_ROI_{horizon_str}'; prob_col = f'Prob_Gain_{horizon_str}'
    if roi_col in rules_df.columns and prob_col in rules_df.columns:
        rules_df.loc[(rules_df[roi_col] > 0.15) & (rules_df[prob_col] > 0.70), 'Action'] = 'Strong Buy'
        rules_df.loc[(rules_df[roi_col] > 0.05) & (rules_df[prob_col] > 0.60), 'Action'] = 'Buy'
        rules_df.loc[(rules_df[roi_col] < -0.15) & (rules_df[prob_col] < 0.30), 'Action'] = 'Strong Sell'
        rules_df.loc[(rules_df[roi_col] < -0.05) & (rules_df[prob_col] < 0.40), 'Action'] = 'Sell'
        
    actionable_rules = rules_df[rules_df['Action'].str.contains('Buy|Sell')].copy()
    if actionable_rules.empty:
        console.print("[yellow]No strong 'Buy' or 'Sell' signals were identified in the simulation.[/yellow]")
        return
        
    path_horizons = rules_df.attrs.get('path_horizons', [])
    
    # Setup columns
    display_cols = ['Action', 'Signal_Feature', 'Optimal_Lag', 'Signal_Interval']
    gradient_cols = []
    for horizon in path_horizons:
        roi_h, prob_h, vol_h = f'Est_ROI_{horizon}', f'Prob_Gain_{horizon}', f'Est_Volatility_{horizon}'
        if roi_h in actionable_rules.columns: display_cols.append(roi_h); gradient_cols.append(roi_h)
        if prob_h in actionable_rules.columns: display_cols.append(prob_h)
        if show_volatility and vol_h in actionable_rules.columns: display_cols.append(vol_h); gradient_cols.append(vol_h)
        
    display_df = actionable_rules[display_cols].sort_values(by=roi_col, ascending=False)
    
    table = Table(title=f"Predictive Rulebook up to {horizon_str}", show_header=True, header_style="bold cyan")
    
    # Add columns to table
    renamed_cols = {
        'Signal_Feature': 'Signal', 'Optimal_Lag': 'Signal Lag', 'Signal_Interval': 'Signal Value',
        **{col: col.replace('Est_ROI_', 'ROI @ ').replace('Prob_Gain_', 'Prob. Gain @ ').replace('Est_Volatility_', 'Est. Vol. @ ') for col in display_cols}
    }
    for col in display_df.columns:
        table.add_column(renamed_cols.get(col, col))
        
    # Color mapping
    cmap = plt.get_cmap('RdYlGn')
    numeric_df = display_df[gradient_cols].apply(pd.to_numeric, errors='coerce')
    vmin, vmax = -max(abs(numeric_df.min().min()), abs(numeric_df.max().max())), max(abs(numeric_df.min().min()), abs(numeric_df.max().max()))
    norm = plt.Normalize(vmin, vmax)
    
    for _, row in display_df.iterrows():
        row_values = []
        for col_name, value in row.items():
            display_val = str(value)
            cell_style = ""
            if col_name in gradient_cols and pd.notna(value):
                hex_color = matplotlib.colors.to_hex(cmap(norm(value)))
                cell_style = f"on {hex_color}"
            
            if 'Est_ROI_' in col_name and pd.notna(value): display_val = f"{value:+.1%}"
            elif 'Prob_Gain_' in col_name and pd.notna(value): display_val = f"{value:.1%}"
            elif 'Est_Volatility_' in col_name and pd.notna(value): display_val = f"{value:.2%}"
            elif 'Signal_Interval' in col_name and isinstance(value, tuple): display_val = f"{value[0]:.5f} to {value[1]:.5f}"
                
            row_values.append(f"[{cell_style}]{display_val}[/]")
            
        table.add_row(*row_values)
        
    console.print(table)


# --- Historical Display Function (CLI Version) ---
def show_historical_report(report_df: pd.DataFrame, horizon_str: str, show_volatility: bool = False):
    if report_df.empty:
        console.print("[yellow]No historically high-certainty signals were found for this scenario.[/yellow]")
        return

    df = report_df.rename(columns={'Signal_Feature': 'Signal', 'Signal_Interval': 'Signal Value'})
    horizon_val = int(horizon_str.replace('M', ''))
    
    # Defensively filter columns
    cols_to_keep = ['Signal', 'Signal Value', 'Occurrences', 'Hist. Prob. of Gain']
    performance_cols = sorted([c for c in df.columns if 'Hist. Avg.' in c], key=lambda x: int(re.search(r'(\d+)M', x).group(1)))
    for col in performance_cols:
        match = re.search(r'(\d+)M', col)
        if match and int(match.group(1)) <= horizon_val:
            if 'Vol' in col and not show_volatility: continue
            cols_to_keep.append(col)
    df = df[cols_to_keep]

    table = Table(title=f"Historical Certainty Report for {horizon_str}", show_header=True, header_style="bold magenta")
    for col in df.columns: table.add_column(col)
        
    # Color mapping logic
    cmap = plt.get_cmap('RdYlGn')
    numeric_cols = [c for c in df.columns if 'Hist. Avg' in c]
    all_vals = pd.to_numeric(df[numeric_cols].stack(), errors='coerce').dropna()
    vmin = -max(abs(all_vals.min()), abs(all_vals.max())) if all_vals.min() < 0 else all_vals.min()
    vmax = max(abs(all_vals.min()), abs(all_vals.max())) if all_vals.min() < 0 else all_vals.max()
    norm = plt.Normalize(vmin, vmax)

    for _, row in df.iterrows():
        if 'SIGNALS ACTIVE IN:' in str(row['Signal']):
            table.add_row(f"[bold white on black]{row['Signal']}[/]", style="bold white on black", end_section=True)
            continue
        
        row_values = []
        for col_name, value in row.items():
            cell_style = ""
            if col_name in numeric_cols and pd.notna(value):
                hex_color = matplotlib.colors.to_hex(cmap(norm(value)))
                cell_style = f"on {hex_color}"
            
            if 'Signal Value' in col_name and isinstance(value, tuple): display_val = f"from {value[0]:.4f} to {value[1]:.4f}"
            elif 'Prob. of Gain' in col_name and pd.notna(value): display_val = f"{value:.1%}"
            elif 'ROI' in col_name and pd.notna(value): display_val = f"{value:+.1%}"
            elif 'Vol' in col_name and pd.notna(value): display_val = f"{value:.2%}"
            elif 'Occurrences' in col_name and pd.notna(value): display_val = f"{value:.0f}"
            else: display_val = str(value)
            
            row_values.append(f"[{cell_style}]{display_val}[/]")
        table.add_row(*row_values)
        
    console.print(table)
```

---

### **Phase 4: Orchestrate in `main.py`**

Finally, we update the main entry point to call our new engine and display functions, completing the logic flow.

*   **File:** `src/VPCLI/main.py`
*   **Action:** Replace the entire file content with the following.

```python:src/VPCLI/main.py
import typer
from rich.console import Console
from rich.prompt import Prompt
from .core import constants
from . import engine, display

app = typer.Typer(
    name="strategy-report",
    help="A CLI for generating hybrid predictive and historical investment strategies.",
    add_completion=False,
)

console = Console()

@app.command(help="Run the interactive strategy report generator.")
def run():
    """
    The main interactive command that guides the user through selecting a
    company and investment horizon, then delegates to the appropriate engine.
    """
    console.print("\n[bold cyan]Welcome to the Financial Strategy Engine![/bold cyan]\n")

    # --- 1. Company Selection ---
    company_prompt_text = "\n".join([f"  [bold yellow]{key}[/bold yellow]: {name}" for key, name in constants.COMPANIES.items()])
    console.print("[bold]First, select a company to investigate:[/bold]")
    console.print(company_prompt_text)
    company_idx = Prompt.ask("Enter selection", choices=list(constants.COMPANIES.keys()), default="2")
    selected_company_name = constants.COMPANIES[company_idx]
    selected_ticker = constants.COMPANY_TICKERS[company_idx]

    # --- 2. Horizon Selection ---
    console.print("\n[bold]Next, select an investment horizon:[/bold]")
    console.print("\n[bold magenta]--- Predictive Recommendations (1M-36M) ---[/bold magenta]")
    console.print("  " + " | ".join(constants.PREDICTIVE_HORIZONS))
    console.print("\n[bold magenta]--- Historical Certainty Analysis (48M-84M) ---[/bold magenta]")
    console.print("  " + " | ".join(constants.HISTORICAL_HORIZONS))
    selected_horizon = Prompt.ask("\nEnter horizon", choices=constants.ALL_HORIZONS, default="12M")

    # --- 3. Orchestration (Final Implementation) ---
    console.print("\n" + "="*50)
    console.print(f"[bold]Generating report for {selected_company_name} | {selected_horizon}...[/bold]")
    console.print("="*50 + "\n")
    
    show_volatility = Prompt.ask("Include volatility estimates in the report?", choices=["y", "n"], default="n") == "y"

    try:
        if selected_horizon in constants.PREDICTIVE_HORIZONS:
            console.print("[italic yellow]Running predictive simulation...[/italic yellow]\n")
            report_data = engine.get_predictive_report(selected_ticker, selected_horizon, show_volatility)
            display.show_predictive_report(report_data, selected_horizon, show_volatility)
        else:
            console.print("[italic yellow]Fetching historical analysis...[/italic yellow]\n")
            report_data = engine.get_historical_report(selected_ticker, selected_horizon)
            display.show_historical_report(report_data, selected_horizon, show_volatility)
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        console.print("[yellow]Please ensure all required .pkl artifacts are present in the 'assets' directory.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")

if __name__ == "__main__":
    app()
```

---

### **Phase 5: Final Verification & Deployment**

1.  **Place Artifacts:** Make sure all three `.pkl` files (`trained_pipelines.pkl`, `modeling_datasets.pkl`, `historical_certainty_reports.pkl`) are copied from your notebook's `out` directory into the CLI's `assets/` directory.
2.  **Local Test:** Run `strategy-report` in your terminal and test both a predictive horizon (e.g., '12M') and a historical one (e.g., '48M').
3.  **Docker Build:** Once local tests pass, build the final deployment package with `docker build -t vpcli .`.
4.  **Docker Run:** Run the container with `docker run --rm -it vpcli` to test the final, deployed application.