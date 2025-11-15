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