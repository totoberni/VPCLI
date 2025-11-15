import pandas as pd
import re
import matplotlib
import matplotlib.pyplot as plt
from rich.console import Console, Group
from rich.table import Table
from rich import markup
from rich.text import Text
from rich.rule import Rule

console = Console()

# --- Predictive Display Function (Unchanged) ---
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
    
    renamed_cols = {
        'Signal_Feature': 'Signal', 'Optimal_Lag': 'Signal Lag', 'Signal_Interval': 'Signal Value',
        **{col: col.replace('Est_ROI_', 'ROI @ ').replace('Prob_Gain_', 'Prob. Gain @ ').replace('Est_Volatility_', 'Est. Vol. @ ') for col in display_cols}
    }
    for col in display_df.columns:
        table.add_column(renamed_cols.get(col, col))
        
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
            
            if cell_style:
                row_values.append(f"[{cell_style}]{display_val}[/]")
            else:
                row_values.append(markup.escape(display_val))
            
        table.add_row(*row_values)
        
    console.print(table)


# --- STATELESS HISTORICAL PAGE RENDERER ---
def render_historical_page(
    page_df: pd.DataFrame,
    page_title: str,
    page_num: int,
    total_pages: int,
    horizon_str: str,
    show_volatility: bool = False
) -> Group:
    """
    Constructs a renderable Group for a single page of a historical report.
    This function is now stateless and does not print to the console.
    """
    title_rule = Rule(f"[bold white on black] {page_title} [/]", style="magenta")

    df = page_df.rename(columns={'Signal_Feature': 'Signal', 'Signal_Interval': 'Signal Value'})
    horizon_val = int(horizon_str.replace('M', ''))
    
    cols_to_keep = ['Signal', 'Signal Value', 'Occurrences', 'Hist. Prob. of Gain']
    performance_cols = sorted(
        [c for c in df.columns if 'Hist. Avg.' in c],
        key=lambda x: int(re.search(r'(\d+)M', x).group(1)) if re.search(r'(\d+)M', x) else 0
    )
    for col in performance_cols:
        match = re.search(r'(\d+)M', col)
        if match and int(match.group(1)) <= horizon_val:
            if 'Vol' in col and not show_volatility: continue
            cols_to_keep.append(col)
    
    df = df[[col for col in cols_to_keep if col in df.columns]]

    table = Table(show_header=True, header_style="bold magenta")
    for col in df.columns: table.add_column(col)
        
    cmap = plt.get_cmap('RdYlGn')
    numeric_cols = [c for c in df.columns if 'Hist. Avg' in c]
    all_vals = pd.to_numeric(df[numeric_cols].stack(), errors='coerce').dropna()
    if not all_vals.empty:
        vmin = -max(abs(all_vals.min()), abs(all_vals.max())) if all_vals.min() < 0 else all_vals.min()
        vmax = max(abs(all_vals.min()), abs(all_vals.max())) if all_vals.min() < 0 else all_vals.max()
        norm = plt.Normalize(vmin, vmax)

    for _, row in df.iterrows():
        row_values = []
        for col_name, value in row.items():
            cell_style, display_val = "", str(value)
            if numeric_cols and col_name in numeric_cols and pd.notna(value):
                hex_color = matplotlib.colors.to_hex(cmap(norm(value)))
                cell_style = f"on {hex_color}"
            
            if 'Signal Value' in col_name and isinstance(value, tuple): display_val = f"from {value[0]:.4f} to {value[1]:.4f}"
            elif 'Prob. of Gain' in col_name and pd.notna(value): display_val = f"{value:.1%}"
            elif 'ROI' in col_name and pd.notna(value): display_val = f"{value:+.1%}"
            elif 'Vol' in col_name and pd.notna(value): display_val = f"{value:.2%}"
            elif 'Occurrences' in col_name and pd.notna(value): display_val = f"{value:.0f}"
            
            row_values.append(f"[{cell_style}]{markup.escape(display_val)}[/]" if cell_style else markup.escape(display_val))
        table.add_row(*row_values)
    
    nav_text = Text(f"\nPage {page_num} of {total_pages}. Use A/D or ←/→ to navigate, Q to quit.", justify="center")
    
    return Group(title_rule, table, nav_text)