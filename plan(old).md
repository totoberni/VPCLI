# Financial Sentiment EDA and Correlation Analysis: Project Roadmap

## 1. Project Overview

**Objective:** To explore, clean, and analyze financial sentiment datasets for Apple (AAPL), NVIDIA (NVDA), and Google (GOOGL). The primary goal is to identify which sentiment and NLP-derived features are most correlated with stock price movements.

**End Goal:** The insights and cleaned data from this analysis will serve as the foundation for feature engineering and selection for a series of company-specific XGBoost models designed to predict daily stock price changes.

**Data Assets:**
*   `AAPL_dataset.csv`, `NVDA_dataset.csv`, `GOOGL_dataset.csv`: Daily sentiment and NLP features.
*   `final_company_dfs.pkl`: A pickle file containing the raw company sentiment dataframes in a dictionary.
*   `Stockprices.csv`: Historical daily stock data (OHLCV) for all three companies.

---

## 2. Phase 1: Setup and Data Unification

This phase focuses on creating clean, standardized, and unified datasets. The process has been refined into a multi-step workflow to handle different data types and missing value patterns robustly.

**2.1. Initial Setup & Verification:**
*   **Action:** Centralize all library installations (`pandas`, `numpy`, `xgboost`, etc.) in the first cell, followed by a mandatory kernel restart.
*   **Action:** Centralize all library imports in the second cell.
*   **Action:** Implement a robust, multi-layer verification script to:
    1.  Confirm NVIDIA driver access via the `nvidia-smi` command.
    2.  Check that the installed `xgboost` package was built with CUDA support.
    3.  Run a small test `fit()` command to confirm the `xgboost` library can successfully initialize the GPU at runtime.

**2.2. Data Loading and Checkpointing:**
*   **Action:** Load the company sentiment data from `final_company_dfs.pkl` and stock price data from `Stockprices.csv`.
*   **Action:** Create separate checkpoint cells to export the final cleaned and imputed dataframe dictionaries (`company_dfs_final` and `stock_dfs_final`) as `.pkl` files.
*   **Action:** Implement a "resume" cell at the beginning of Phase 2 to load these checkpointed `.pkl` files, allowing the user to bypass the entire cleaning pipeline.

**2.3. Sentiment Dataset Imputation Strategy:**
*   A two-tiered imputation strategy is applied:
    *   **Tier 1 (Simple Imputation):** For `article_volume` and `market_average_sentiment`, `0` values are replaced with `NaN` and imputed using `ffill` and `bfill`.
    *   **Tier 2 (Advanced Imputation):** For `average_news_sentiment` and `mspr`, a sophisticated model-based approach is used.

**2.4. Advanced Imputation Workflow:**
*   **Action: Hyperparameter Tuning:** A `tune_xgboost_hyperparameters` function uses `RandomizedSearchCV` to find the optimal hyperparameters for the `XGBRegressor` estimator for each specific company.
*   **Action: Master Imputation Loop:** A master loop iterates through each company, creating a dedicated, uniquely tuned `IterativeImputer` with early stopping to fill missing values.

**2.5. Stock Price Dataset Cleaning and Standardization:**
*   **Action: Clean and Split:** A function cleans the string-based price and volume data into numeric types. The wide-format `stocks_df_raw` is then split into three separate, standardized DataFrames.
*   **Action: Align and Impute:** Each stock DataFrame is re-indexed to a master date range (`2018-01-01` to `2024-12-31`). The resulting `NaN` values for non-trading days are imputed using `ffill` and `bfill`.

---

## 3. Phase 2: Exploratory Data Analysis (EDA)

This phase visually explores feature characteristics and their relationships with stock prices. The strategy is repeated for each company (`AAPL`, `NVDA`, `GOOGL`).

**3.1. Univariate Analysis: Understanding Feature Distributions**
*   **Objective:** To understand the statistical properties of each key feature.
*   **Actions:** For each feature, generate a histogram, a box plot, and descriptive statistics.

**3.2. Bivariate Time-Series Analysis: Finding Predictive Relationships**
*   **Objective:** To visually inspect how sentiment features co-move with stock price and volatility over time.
*   **Methodology:** Merge the sentiment and stock dataframes for each company. Generate a series of time-series plots.

---

## 4. Phase 3: Multi-Horizon Correlation and Predictive Power Analysis

This phase moves beyond visual EDA to **quantify** the predictive power of sentiment features across multiple investment time horizons. The goal is to identify the most promising features, their optimal time lags, and the timeframes over which they are most effective.

**4.1. Engineering a Spectrum of Predictive Targets:**
*   **Objective:** To create a rich set of target variables that capture future returns and volatility over short, medium, and long-term windows.
*   **Actions:**
    1.  **Consolidate Data:** Merge the final sentiment and stock data for each company into a unified `eda_dfs` dictionary.
    2.  **Define Horizons:** Establish a set of lookahead periods: `[1D, 1M, 3M, 6M, 12M, 18M, ..., 84M]`.
    3.  **Engineer Return Targets:** For each horizon `n`, create a `Future_Return_{n}` column, calculating the total percentage return between day `t` and day `t + n_months`.
    4.  **Engineer Volatility Targets:** For each horizon `n`, create a `Future_Volatility_{n}` column, calculating the rolling standard deviation of `daily_return` over the next `n` months.

**4.2. Systematic Multi-Horizon Lag Analysis:**
*   **Objective:** To discover the optimal predictive lag for every combination of sentiment feature and multi-horizon target.
*   **Methodology:**
    1.  **Define Lag Function:** Create a `calculate_lag_correlations` function that iterates through a range of time lags (e.g., `t-5` to `t+5` days), calculating both **Pearson** and **Spearman** correlation for each feature against the full suite of future targets.
*   **Action:** Run this systematic analysis for each company (`AAPL`, `NVDA`, `GOOGL`).

**4.3. "Meta-Analysis" Visualization and Interpretation:**
*   **Objective:** To distill the complex correlation results into clear, high-level summary visualizations.
*   **Methodology:**
    1.  **Define Heatmap Function:** Create a `plot_peak_correlation_heatmap` function. This function finds the maximum absolute correlation for each feature-horizon pair across all tested lags and visualizes the result as a heatmap. Crucially, it must safely handle all-NaN groups to prevent errors.
    2.  **Define Lag Curve Function:** Create a `plot_lag_curves` function to generate "drill-down" plots for specific feature/horizon combinations.
*   **Actions:**
    1.  **Generate Peak Correlation Heatmaps:** For each company, use the heatmap function to generate two primary visualizations (one for Future Returns, one for Future Volatility). Each cell will be annotated with the peak correlation value and the optimal lag at which it occurred (e.g., "-0.89 @ -5d").
    2.  **Drill Down with Lag Plots:** Based on the most interesting signals from the heatmaps, use the lag curve function to generate detailed plots for specific combinations (e.g., `10-Q_sentiment` vs. long-term returns).

**4.4. Synthesis and Action Plan:**
*   **Objective:** To summarize the multi-horizon findings into a precise feature engineering blueprint.
*   **Action:** For each company, create a summary markdown cell structured by investment horizon (Short-Term, Medium-Term, Long-Term, Volatility). This will list the most predictive features, their peak correlation scores, their optimal lags, and a brief interpretation of the strategic implication.

---

## 5. Phase 5: Stakeholder Visualization & Insight Validation

**Objective:** To translate the quantitative findings from the correlation analysis (Phase 3) into compelling, easy-to-understand visualizations for business stakeholders (hedge fund managers). This phase will validate the identified signals and build the business case for the subsequent feature engineering phase.

**5.1. Core Logic: Isolate Filing Events**
*   **Objective:** The sentiment data is forward-filled. We must first identify the exact dates on which new filings were released.
*   **Action:** Create a reusable Python function `get_filing_dates(df, sentiment_column)`.
*   **Methodology:** This function will take a company DataFrame and a sentiment column name (e.g., `'10-Q_sentiment'`) as input. It will identify event dates by finding where the sentiment score `.diff() != 0`. It must return a new DataFrame containing only the filing dates and their corresponding non-NaN sentiment scores.

**5.2. Visualization Part 1: Event-Study Charts for Short-Term Price Impact**
*   **Objective:** Visualize the average stock price trajectory around filing events, segmented by sentiment, to answer: "How does the market react in the days immediately following a filing, based on its sentiment?"
*   **Actions:**
    1.  **Define a master plotting function:** `plot_event_study(company_df, filing_events_df, filing_type, stock_price_col='Close')`.
    2.  **Data Preparation (within function):** For each filing event date `t`:
        *   Slice a 21-day window (`t-10` to `t+10`) from the main `company_df`.
        *   **Normalize Price:** Normalize the stock price column by dividing all values in the window by the price at `t-1` and multiplying by 100. This converts the price to a common scale showing percentage change relative to the day before the event.
    3.  **Sentiment Grouping (within function):**
        *   Use `pd.qcut` to divide all filing events into three quantiles based on their sentiment scores: 'Low Sentiment' (bottom 33%), 'Mid Sentiment' (middle 34%), and 'High Sentiment' (top 33%).
    4.  **Aggregation (within function):**
        *   For each sentiment group, calculate the `mean` of all the normalized price series at each day from -10 to +10. This produces the average trajectory for each group.
    5.  **Visualization (within function):**
        *   Plot the three average trajectories on a single chart using `matplotlib` or `seaborn`.
        *   Use a clear, intuitive color scheme (e.g., Red for Low, Gray for Mid, Green for High).
        *   Add a vertical dashed line at `x=0` labeled "Filing Date".
        *   Set a descriptive title, legend, and axis labels (`Days Relative to Filing`, `Normalized Price (Day -1 = 100)`).
*   **Execution:** Create a loop to execute this plotting function for each company (AAPL, NVDA, GOOGL) and each filing type (10-K, 10-Q, 8-K), generating 9 distinct event-study charts.

**5.3. Visualization Part 2: Scatter Plots for Long-Term ROI & Volatility Impact**
*   **Objective:** Visually confirm the strong, long-term correlations discovered in Phase 3 to answer: "How predictive is filing sentiment for multi-year returns and volatility?"
*   **Actions:**
    1.  **Identify Key Relationships:** From the Phase 3 markdown summaries, codify the most powerful feature/target pairs into a configuration list. Example: `[('NVDA', '10-Q_sentiment', 'Future_Return_72M'), ('GOOGL', 'mspr', 'Future_Volatility_48M')]`.
    2.  **Define a master plotting function:** `plot_correlation_scatter(company_df, filing_events_df, feature_col, target_col)`.
    3.  **Data Preparation (within function):** Use the `filing_events_df` (from 5.1) to get the specific dates and sentiment scores. For each of these dates, retrieve the corresponding future outcome value (e.g., `Future_Return_72M`) from the main `company_df`.
    4.  **Visualization (within function):**
        *   Use `seaborn.regplot` to create a scatter plot of the `feature_col` (sentiment) vs. the `target_col` (outcome). This function automatically includes the regression line and a confidence interval, which is ideal for this use case.
        *   Set a clear title that includes the company, feature, and target (e.g., "NVDA: 10-Q Sentiment vs. 72-Month Future Return").
*   **Execution:** Iterate through the list of key relationships, calling the plotting function for each one.

**5.4. Synthesis and Final Reporting**
*   **Action:** Create a final summary markdown cell in the `AnalystSentimentEDA.ipynb` notebook.
*   **Content:** This cell will display the most impactful charts generated in this phase. Each chart will be accompanied by a concise, business-friendly takeaway that directly references the visualization and its strategic implication (e.g., "As seen below, pessimistic 10-Q reports for NVDA have historically preceded periods of significant long-term outperformance, highlighting a powerful contrarian signal.").

## 6. Phase 6 (Revised): The Hybrid Predictive & Historical Strategy Engine

**Objective:** To finalize the proof-of-concept by building a complete, two-part statistical machine. This engine will provide forward-looking probabilistic forecasts for short-to-medium term horizons and backward-looking historical certainty analysis for long-term horizons. The final artifacts—trained models and historical reports—will be designed for direct use by the Phase 7 CLI tool.

---

### **6.1: Engine Setup and Data Ingestion**

*   **Objective:** To create a clean, isolated starting point for the predictive engine by loading a single, pre-merged data asset.
*   **Actions:**
    1.  **Load Pre-Merged Data:** Load a single checkpoint file, `eda_dfs_merged.pkl`. This file is expected to contain a dictionary where keys are company tickers and values are the corresponding DataFrames with sentiment and stock data already merged.
    2.  **Establish Engine Data Source:** The loaded dictionary will be designated as the `engine_dfs`, serving as the master data source for all subsequent steps within this phase. This ensures the engine starts from a consistent, consolidated state.

---

### **6.2: Dynamic, In-Engine Feature Engineering**

*   **Objective:** To programmatically create the multi-horizon target variables required for model training.
*   **Actions:**
    1.  **Define Target Engineering Function:** Create a function `engineer_predictive_targets(df, horizons_config)`.
    2.  **Methodology:** This function will take a company's dataframe and a horizon configuration dictionary (e.g., `{'3M': 63, '6M': 126, ...}`). For each horizon, it will calculate and append two new columns:
        *   `Future_Return_{horizon}`: The total percentage return over the future period.
        *   `Future_Volatility_{horizon}`: The rolling standard deviation of daily returns over the future period.
    3.  **Execute Engineering:** Iterate through each dataframe in the `engine_dfs` dictionary and apply this function to create the fully engineered datasets. A new checkpoint, `engine_dfs_engineered.pkl`, will be exported after this step.

---

### **6.3: In-Engine Correlation Analysis for Feature Selection**

*   **Objective:** To systematically quantify the predictive power of all potential signals to inform feature selection for the predictive models. This is a crucial data-driven step to reduce model complexity and improve performance.
*   **Actions:**
    1.  **Define Correlation Function:** Re-implement the `calculate_lag_correlations` function within this phase.
    2.  **Execute Analysis:** For each company, run a comprehensive lead-lag correlation analysis between all potential sentiment/NLP features and all engineered future targets (`Future_Return_*`, `Future_Volatility_*`).
    3.  **Aggregate Results:** Combine the correlation results from all companies into a single `master_corr_results` DataFrame. This dataframe is a metadata asset that will be used to select the best features for each model.

---

### **6.4: The Predictive Core - Adaptive Modeling Dataset Construction (Revised)**

*   **Objective:** To construct modeling datasets that adapt to the length of the forecast horizon. For short horizons where a true out-of-time validation set is feasible, it will produce a train/validation split. For long horizons where it is not, it will produce a single, unified training set to maximize the data available for model training.
*   **Actions:**
    1.  **Define Adaptive Feature Vector Builder:** Create a master function `build_modeling_dataset_adaptive(df, ticker, horizon, corr_results, top_k_features=10)`.
    2.  **Methodology:** This function will:
        *   Consult `corr_results` to identify the `top_k_features` (and their optimal lags) specifically for the given `ticker` and `horizon`.
        *   Construct the feature matrix `X` by creating the correctly lagged feature columns.
        *   Isolate the target vectors `Y_return` and `Y_volatility`.
        *   Create the full `modeling_df` by dropping any rows with `NaN` values resulting from the feature lagging.
        *   **Implement Adaptive Split Logic:**
            *   Define a fixed chronological split date (e.g., `TRAIN_END_DATE = '2022-12-31'`).
            *   Attempt to create a `val_df` using data after this date.
            *   **Crucially, check if this `val_df` contains any non-NaN values for the specific `horizon`'s target columns.** Due to the long lookahead period, these values will be `NaN` for later dates in the dataframe.
            *   **If a valid validation set exists (Short Horizons):** The function will return the six distinct data components as before: `(features, X_train, Y_r_train, Y_v_train, X_val, Y_r_val, Y_v_val)`.
            *   **If a valid validation set does NOT exist (Long Horizons):** The function will not split the data. It will use the **entire** `modeling_df` as the training set and return `None` for all validation components. The return signature will be: `(features, X_entire, Y_r_entire, Y_v_entire, None, None, None)`. This ensures that for long-horizon models, we train on the maximum available historical data.
    3.  **Execute Construction Loop:** The main loop for building datasets will iterate through all tickers and horizons, calling the new adaptive function. It will store the results, correctly handling cases where the validation set is `None`.


---

*(Steps 6.1 through 6.4 are considered complete and have successfully produced the `engine_dfs_engineered.pkl`, `master_corr_results.pkl`, and `modeling_datasets` artifacts.)*


### **Part A: The Predictive Engine (Horizons 1M-36M)**

This part focuses on refining and finalizing the XGBoost model training for horizons where out-of-time validation is feasible.

#### **6.5: High-Precision Model Training & Consolidated Reporting**

*   **Objective:** To retrain the short-to-medium term models with a more exhaustive hyperparameter search and to generate a single, unified HTML performance report for all trained models.
*   **Actions:**
    1.  **Refine Tuning in `tune_and_train`:**
        *   **Methodology:** In the `objective` function (cell `6.5.0`), significantly expand the hyperparameter search space. Increase `n_trials` in the `study.optimize` call from 30 to a higher number (e.g., 50 or 75) to allow Optuna to explore more combinations and find a more precise set of optimal parameters. Since this now only runs for horizons up to 36M, the extra computation time is justified.
    2.  **Overhaul `test_and_report_adaptive` for Unified Reporting:**
        *   **Methodology:** This function (cell `6.5.1`) will be refactored. Instead of writing a file inside the loop, it will now **return a dictionary** containing the `plotly` figure object and the HTML metrics table string for a given model.
        *   Example return: `{'figure': fig, 'metrics_html': '<table>...'}`. It will return `None` for long-horizon models where no report is generated.
    3.  **Execute a Refined Master Training & Reporting Loop:**
        *   **Methodology:** The master loops (cells `6.5.2` and `6.5.3`) will be combined. A new list, `html_report_components`, will be initialized.
        *   The loop will iterate from 1M to 36M horizons. Inside the loop, it will first call the refined `tune_and_train` and then immediately call the refactored `test_and_report_adaptive`.
        *   If the reporting function returns a result, the figure and HTML string will be appended to the `html_report_components` list.
    4.  **Generate Single Master Report:**
        *   **Methodology:** After the loop completes, a new cell will iterate through the `html_report_components` list. It will write a master HTML file, sequentially appending the HTML representation of each figure and each metrics table. This creates a single, comprehensive document for easy review.
    5.  **Export Final Predictive Models:** The `trained_pipelines` dictionary, now containing only the highly-tuned models for horizons up to 36M, will be saved to `trained_pipelines.pkl`.

---

### **Part B: The Historical Certainty Engine (Horizons 48M-84M)**

This part implements the new "Phase 6B" analysis, focusing on scenarios where the predictive model could not be trained due to a lack of negative outcomes.

#### **6.6: Historical Outcome Analysis**

*   **Objective:** To build and execute an engine that analyzes historical data for long-term horizons to identify and quantify highly deterministic signal-outcome relationships.
*   **Actions:**
    1.  **Define `analyze_historical_outcomes` Engine:**
        *   **Methodology:** Create a new function that accepts a ticker, horizon, the full `engine_dfs_engineered`, and `master_corr_results`.
        *   It will identify the most potent feature for that `(ticker, horizon)` pair from the correlation results.
        *   It will then segment that feature's historical values into deciles using `pd.qcut`.
        *   **Crucially, for each decile**, it will group the data and calculate not just the outcome for the target horizon (e.g., 48M), but for a **range of horizons** (`3M, 6M, 12M, 24M... up to the target horizon`). This addresses the "month-by-month" reporting goal by showing the historical performance *path*.
        *   The output will be a DataFrame where each row represents a signal decile, and columns include `Historical_Prob_Gain`, `Avg_ROI_at_12M`, `Avg_ROI_at_24M`, `Avg_ROI_at_48M`, etc.
    2.  **Execute Historical Analysis Loop:**
        *   **Methodology:** Create a new master loop that iterates through all tickers for horizons from 48M to 84M.
        *   Inside the loop, call `analyze_historical_outcomes`.
        *   Store the resulting analysis DataFrames in a new dictionary: `historical_certainty_reports`.
    3.  **Export Historical Reports:** Serialize the complete `historical_certainty_reports` dictionary to a new artifact: `historical_certainty_reports.pkl`.

---

### **6.7: Final Unified Strategy Interface (Proof-of-Concept)**

*   **Objective:** To create a single, user-facing function that demonstrates the final hybrid logic, serving as the direct blueprint for the Phase 7 CLI tool.
*   **Actions:**
    1.  **Define Final `get_strategy_report` Function:**
        *   **Methodology:** This master function will take a `ticker` and `horizon` as input.
        *   It will contain a simple conditional check: `if horizon <= 36M:`.
        *   **Predictive Path (<= 36M):** It will call `simulate_investment_rules` using the loaded `trained_pipelines` and then pass the result to a `display_predictive_rulebook` function for formatted console output.
        *   **Historical Path (>= 48M):** It will retrieve the pre-computed report from the `historical_certainty_reports` dictionary and pass it to a new `display_historical_report` function. This new display function will filter for the most compelling results (e.g., deciles with 100% historical success) and print a formatted table showing the performance path (Avg ROI at 12M, 24M, 48M, etc.).
    2.  **Demonstrate Usage:** The final cell of the notebook will contain example calls to this master function for both short and long horizons, proving that the unified logic works and that both artifacts (`trained_pipelines.pkl` and `historical_certainty_reports.pkl`) can be used to deliver a complete strategic overview.
        ```python
        # Show a predictive report
        get_strategy_report(ticker='NVDA', horizon='12M') 
        
        # Show a historical certainty report
        get_strategy_report(ticker='NVDA', horizon='48M')
        ```
---

## 7. Phase 7 (Corrected): The Live CLI Inference Engine

**Objective:** To build an interactive command-line tool that is the direct, user-facing implementation of the hybrid strategy engine developed in Phase 6. The tool will load the trained models and run predictive simulations in real-time for short horizons, and load pre-computed reports for long-horizon historical analysis.

**Core Principle:** The CLI performs real-time **predictive simulation** using the pre-trained models. The heavy lifting of model training and the exhaustive calculation of historical statistics are done offline in the notebook.

---

### **7.1: Final Asset Management**

This phase depends on the primary artifacts generated during the offline Phase 6 pipeline. The CLI will be packaged with these assets.

1.  **`trained_pipelines.pkl`**: **(Required)** A dictionary containing the actual, trained XGBoost model suites for all short-to-medium term horizons (1M-36M). This is the "live brain" of the predictive engine.
2.  **`historical_certainty_reports.pkl`**: **(Required)** A dictionary containing the pre-calculated historical analysis reports for all long-term horizons (48M-84M).
3.  **`modeling_datasets.pkl`**: **(Required)** A dictionary containing the training data splits. The CLI specifically needs the `X_train` reference data from this file to establish the baseline (median) values for running simulations.
4.  **`master_performance_report.html`**: **(Optional)** The consolidated HTML file containing validation metrics for all trained models, for diagnostic purposes.

---

### **7.2: CLI Project Scaffolding**

*   **Objective:** To set up a standard, distributable Python project structure.
*   **Actions:**
    1.  Create a project directory: `financial_strategist_cli/`.
    2.  Create subdirectories: `src/` for application code and `assets/` to store the `.pkl` and `.html` artifact files.
    3.  Set up a `pyproject.toml` file to manage project metadata and dependencies.
    4.  **Define Dependencies:** `pandas`, `xgboost`, `scikit-learn` (for model loading and predicting), `typer` (for the CLI interface), and `rich` (for polished console output).

---

### **7.3: The Application Core**

The application logic will be cleanly separated into a main entrypoint, an engine for running simulations, and a display layer for formatting.

*   **`main.py` (The Orchestrator):**
    *   **Objective:** The main entry point that handles user input and orchestrates the application flow.
    *   **Methodology:**
        1.  **Asset Loading:** At script startup, load all required artifacts (`trained_pipelines.pkl`, `historical_certainty_reports.pkl`, `modeling_datasets.pkl`) into memory. This is a one-time "warm-up" cost.
        2.  **CLI Command Definition:** Using `typer`, define the main command `get_strategy(ticker: str, horizon: str, show_model_report: bool = False)`. Use `typer.Option` with autocompletion to guide the user to select from valid tickers and horizons.
        3.  **Hybrid Logic Execution:**
            *   Based on the `horizon`, determine whether to enter the predictive or historical path.
            *   **Predictive Path (<= 36M):**
                *   Retrieve the specific `pipeline` from the `trained_pipelines` dictionary.
                *   Retrieve the corresponding `X_train_ref` from the `modeling_datasets` dictionary.
                *   Call `engine.run_predictive_simulation(pipeline, X_train_ref)` to get the simulation results DataFrame.
                *   Pass this DataFrame to `display.display_predictive_rulebook()`.
            *   **Historical Path (>= 48M):**
                *   Call `engine.get_historical_report(ticker, horizon)` to retrieve the pre-computed DataFrame.
                *   Pass this DataFrame to `display.display_historical_report()`.
        4.  **Optional Report:** If `--show-model-report` is passed, use the `webbrowser` module to open the local `master_performance_report.html` file.

*   **`engine.py` (The Live Engine):**
    *   **Objective:** To house the core logic for running simulations and retrieving data.
    *   **Functions:**
        *   `run_predictive_simulation(pipeline, X_train_ref)`: This function is a direct port of the `simulate_investment_rules` function from the Phase 6 notebook. It takes the trained models and reference data, runs the scenario simulations in real-time, and returns the resulting DataFrame. This is where the month-by-month "performance path" predictions will be generated.
        *   `get_historical_report(ticker, horizon, historical_data)`: A simple retriever function that accesses the `historical_certainty_reports` dictionary and returns the relevant pre-computed DataFrame.

*   **`display.py` (The Presentation Layer):**
    *   **Objective:** To format the raw DataFrames from the engine into clean, human-readable console tables using the `rich` library.
    *   **Functions:**
        *   `display_predictive_rulebook(rules_df)`: Formats the simulation results. It will present the "performance path" by showing the predicted ROI at multiple intervals (e.g., Est. ROI at 3M, 6M, 12M...) as separate columns, giving the user a clear view of the investment's expected trajectory under the specified signal condition.
        *   `display_historical_report(report_df)`: Formats the historical analysis. It will filter for the most compelling results (e.g., deciles with 100% historical success) and similarly present the "performance path" with columns like `Avg. Historical ROI (at 12M)`, `Avg. Historical ROI (at 24M)`, etc.

---

### **7.4: Packaging and Distribution**

*   **Objective:** To package the tool into a single, easily installable file for the client.
*   **Actions:**
    1.  Configure `pyproject.toml` to define a console script entry point (e.g., `strategy-report = src.main:app`) and to include the `assets` directory as package data.
    2.  Use standard tools (`pip`, `build`) to create a distributable wheel (`.whl`) file.
    3.  The client can `pip install strategy_report.whl` and run the tool, which will now have access to the bundled models and data, directly from their terminal:
        ```bash
        strategy-report --ticker "NVDA" --horizon "24M"
        ```