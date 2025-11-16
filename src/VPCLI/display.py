import re
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import ast
from rich import markup
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# Import the new color utility
from .core.colors import get_contrasting_text_color

console = Console()


def show_predictive_report(
    rules_df: pd.DataFrame, horizon_str: str, show_volatility: bool = False
):
    if rules_df.empty:
        console.print(
            "[yellow]No predictive rules could be generated for this scenario.[/yellow]"
        )
        return

    # --- Refined Rule & Sorting Logic ---
    roi_col = f"Est_ROI_{horizon_str}"
    prob_col = f"Prob_Gain_{horizon_str}"

    # 1. Calculate Expected Value
    if roi_col in rules_df.columns and prob_col in rules_df.columns:
        rules_df["Expected_Value"] = rules_df[roi_col] * rules_df[prob_col]
    else:
        # If key columns are missing, we can't proceed.
        console.print(
            f"[red]Error: Required columns '{roi_col}' or '{prob_col}' not found.[/red]"
        )
        return

    # 2. Assign Actions based on the new refined rule set
    rules_df["Action"] = "Neutral"  # Start with a neutral default

    # Define conditions for each action
    # Note: ROI and Probabilities are floats (e.g., 0.1 for 10%)
    # Expected Value thresholds are adjusted accordingly (e.g., 5 for 500%)
    
    # Strong Buy
    strong_buy_cond = (
        (rules_df["Expected_Value"] > 0.05)  # EV > 5%
        & (rules_df[prob_col] > 0.60)
        & (rules_df[roi_col] > 0.10)
    )
    rules_df.loc[strong_buy_cond, "Action"] = "Strong Buy"

    # Buy
    buy_cond = (
        (rules_df["Expected_Value"] > 0.005) # EV > 0.5%
        & (rules_df[prob_col] > 0.50)
        & (~strong_buy_cond)  # Exclude signals that are already Strong Buy
    )
    rules_df.loc[buy_cond, "Action"] = "Buy"
    
    # Strong Sell
    strong_sell_cond = (
        (rules_df["Expected_Value"] < -0.05) # EV < -5%
        & (rules_df[prob_col] < 0.40)
        & (rules_df[roi_col] < -0.10)
    )
    rules_df.loc[strong_sell_cond, "Action"] = "Strong Sell"

    # Sell
    sell_cond = (
        (rules_df["Expected_Value"] < -0.005) # EV < -0.5%
        & (rules_df[prob_col] < 0.50)
        & (~strong_sell_cond) # Exclude signals that are already Strong Sell
    )
    rules_df.loc[sell_cond, "Action"] = "Sell"

    # Strong Hold
    hold_cond = (
        (rules_df["Expected_Value"].abs() < 0.005) # EV is close to zero
        & (rules_df[roi_col].abs() < 0.05)        # ROI is also small
        & (rules_df[prob_col].between(0.40, 0.60)) # Probability is uncertain
    )
    rules_df.loc[hold_cond, "Action"] = "Strong Hold"
    
    # 3. Sort by the absolute Expected Value to show most impactful signals first
    rules_df["abs_ev"] = rules_df["Expected_Value"].abs()
    rules_df = rules_df.sort_values(by="abs_ev", ascending=False)

    # 4. Filter for only actionable signals (Option A from our plan)
    actionable_rules = rules_df[
        rules_df["Action"].str.contains("Buy|Sell|Strong Hold")
    ].copy()

    if actionable_rules.empty:
        console.print(
            "[yellow]No confident 'Buy', 'Sell', or 'Hold' signals were identified in the simulation.[/yellow]"
        )
        return

    path_horizons = rules_df.attrs.get("path_horizons", [])

    # Setup display columns
    display_cols = ["Action", "Signal_Feature", "Optimal_Lag", "Signal_Interval"]
    gradient_cols = []
    for horizon in path_horizons:
        roi_h, prob_h, vol_h = (
            f"Est_ROI_{horizon}",
            f"Prob_Gain_{horizon}",
            f"Est_Volatility_{horizon}",
        )
        if roi_h in actionable_rules.columns:
            display_cols.append(roi_h)
            gradient_cols.append(roi_h)
        if prob_h in actionable_rules.columns:
            display_cols.append(prob_h)
        if show_volatility and vol_h in actionable_rules.columns:
            display_cols.append(vol_h)
            gradient_cols.append(vol_h)

    # Use the sorted and filtered dataframe for display
    display_df = actionable_rules[display_cols]

    table = Table(
        title=f"Predictive Rulebook up to {horizon_str}",
        show_header=True,
        header_style="bold cyan",
    )

    # CORRECTED and SIMPLIFIED column renaming logic
    for col in display_df.columns:
        header = col.replace("_", " ")
        header = header.replace("Est ROI", "ROI")
        header = header.replace("Prob Gain", "Prob. Gain")
        header = header.replace("Est Volatility", "Est. Vol.")
        table.add_column(header)

    cmap = plt.get_cmap("RdYlGn")
    numeric_df = display_df[gradient_cols].apply(pd.to_numeric, errors="coerce")
    vmin, vmax = -max(
        abs(numeric_df.min().min()), abs(numeric_df.max().max())
    ), max(abs(numeric_df.min().min()), abs(numeric_df.max().max()))
    norm = plt.Normalize(vmin, vmax)

    for _, row in display_df.iterrows():
        row_values = []
        for col_name, value in row.items():
            display_val = str(value)
            cell_style = ""
            if col_name in gradient_cols and pd.notna(value):
                hex_color = matplotlib.colors.to_hex(cmap(norm(value)))
                text_color = get_contrasting_text_color(hex_color)
                cell_style = f"{text_color} on {hex_color}"

            if "Est_ROI_" in col_name and pd.notna(value):
                display_val = f"{value:+.1%}"
            elif "Prob_Gain_" in col_name and pd.notna(value):
                display_val = f"{value:.1%}"
            elif "Est_Volatility_" in col_name and pd.notna(value):
                display_val = f"{value:.2%}"
            elif "Signal_Interval" in col_name and isinstance(value, tuple):
                display_val = f"{value[0]:.5f} to {value[1]:.5f}"

            if cell_style:
                row_values.append(f"[{cell_style}]{display_val}[/]")
            else:
                row_values.append(markup.escape(display_val))

        table.add_row(*row_values)

    console.print(table)


# --- REFACTORED Historical Page Renderer ---
def render_historical_page(
    month_df: pd.DataFrame,
    month: pd.Timestamp,
    page_num: int,
    total_pages: int,
    horizon_str: str,
    show_volatility: bool = False,
):
    """
    Renders a pre-aggregated summary table for signals active in a specific month.
    """
    month_str = month.strftime("%B %Y")
    
    # 1. Define the columns to display
    cols_to_display = [
        "Signal_Feature",
        "Signal_Interval",
        "Occurrences",
        "Win/Loss Ratio",
        f"Hist. Avg. ROI @ {horizon_str}",
        f"Min ROI @ {horizon_str}",
        f"Max ROI @ {horizon_str}",
    ]
    if show_volatility:
        cols_to_display.append(f"Std Dev ROI @ {horizon_str}")
    
    final_cols = [col for col in cols_to_display if col in month_df.columns]
    display_df = month_df[final_cols]
    
    # 2. --- Table Rendering ---
    table = Table(
        title=f"Aggregated Signal Performance for: {month_str}",
        show_header=True,
        header_style="bold magenta",
    )

    renamed_cols = {
        "Signal_Feature": "Signal",
        "Signal_Interval": "Signal Value",
        "Win/Loss Ratio": "Win/Loss (Ratio)",
        f"Std Dev ROI @ {horizon_str}": f"Std Dev @ {horizon_str}",
    }
    for col in display_df.columns:
        header = renamed_cols.get(col, col.replace("_", " "))
        table.add_column(header)
        
    cmap = plt.get_cmap("RdYlGn")
    numeric_cols = [c for c in display_df.columns if "ROI" in c]
    all_vals = pd.to_numeric(display_df[numeric_cols].stack(), errors="coerce").dropna()

    if not all_vals.empty:
        vmax = max(abs(all_vals.min()), abs(all_vals.max()))
        vmin = -vmax
        norm = plt.Normalize(vmin, vmax)

    for _, row in display_df.iterrows():
        row_values = []
        for col_name, value in row.items():
            cell_style, display_val = "", str(value)
            if numeric_cols and col_name in numeric_cols and pd.notna(value):
                hex_color = matplotlib.colors.to_hex(cmap(norm(value)))
                text_color = get_contrasting_text_color(hex_color)
                cell_style = f"{text_color} on {hex_color}"

            # CORRECTED LOGIC: Replicated from the predictive report display function.
            if col_name == "Signal_Interval" and isinstance(value, tuple):
                 if len(value) == 2:
                    display_val = f"from {value[0]:.5f} to {value[1]:.5f}"
                 else:
                    display_val = str(value)
            elif "ROI" in col_name and pd.notna(value):
                display_val = f"{value:+.1%}"
            elif "Std Dev" in col_name and pd.notna(value):
                 display_val = f"{value:.2%}"
            elif "Occurrences" in col_name and pd.notna(value):
                display_val = f"{int(value)}"
            
            if cell_style:
                row_values.append(f"[{cell_style}]{markup.escape(display_val)}[/]")
            else:
                row_values.append(markup.escape(display_val))
        table.add_row(*row_values)
    
    nav_text = Text(
        f"\nPage {page_num} of {total_pages}. Use A/D or < / > to navigate, Q to quit.",
        justify="center",
    )
    
    console.print(table)
    console.print(nav_text)