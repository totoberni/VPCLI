import pandas as pd
import re
import matplotlib
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich import markup
from rich.prompt import Prompt
from rich.rule import Rule

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


# --- REFACTORED HISTORICAL DISPLAY FUNCTION ---
def show_historical_report(report_df: pd.DataFrame, horizon_str: str, show_volatility: bool = False):
    if report_df.empty:
        console.print("[yellow]No historically high-certainty signals were found for this scenario.[/yellow]")
        return

    # --- Step 1: Split DataFrame into Pages ---
    pages = []
    page_titles = []
    current_page_rows = []
    
    for _, row in report_df.iterrows():
        if 'SIGNALS ACTIVE IN:' in str(row['Signal_Feature']):
            if current_page_rows:
                pages.append(pd.DataFrame(current_page_rows))
            current_page_rows = []
            page_titles.append(row['Signal_Feature'])
        else:
            current_page_rows.append(row)
    if current_page_rows:
        pages.append(pd.DataFrame(current_page_rows))

    if not pages:
        console.print("[yellow]No historical signals found to display.[/yellow]")
        return

    # --- Step 2: Start Pagination Loop ---
    current_page_index = 0
    while True:
        console.clear()
        page_df = pages[current_page_index]
        page_title = page_titles[current_page_index]

        # --- Step 3: Render the page (with improved delimiter) ---
        console.print(Rule(f"[bold white on black] {page_title} [/]", style="magenta"))

        df = page_df.rename(columns={'Signal_Feature': 'Signal', 'Signal_Interval': 'Signal Value'})
        horizon_val = int(horizon_str.replace('M', ''))
        
        cols_to_keep = ['Signal', 'Signal Value', 'Occurrences', 'Hist. Prob. of Gain']
        performance_cols = sorted([c for c in df.columns if 'Hist. Avg.' in c], key=lambda x: int(re.search(r'(\d+)M', x).group(1)))
        for col in performance_cols:
            match = re.search(r'(\d+)M', col)
            if match and int(match.group(1)) <= horizon_val:
                if 'Vol' in col and not show_volatility: continue
                cols_to_keep.append(col)
        df = df[cols_to_keep]

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
        
        console.print(table)
        
        # --- Step 4: Navigation Prompt ---
        prompt_text = f"\nPage {current_page_index + 1} of {len(pages)}. "
        choices = []
        if current_page_index > 0: choices.append("P")
        if current_page_index < len(pages) - 1: choices.append("N")
        choices.append("Q")
        
        nav_prompt = " ".join([f"[{c}]" + ("revious" if c=='P' else "ext" if c=='N' else "uit") for c in choices])
        action = Prompt.ask(nav_prompt, choices=[c.lower() for c in choices], default="q").lower()

        if action == 'n': current_page_index += 1
        elif action == 'p': current_page_index -= 1
        elif action == 'q': break