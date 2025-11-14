import typer
from rich.console import Console
from rich.prompt import Prompt

from .core import constants

# Single Typer app instance to manage all commands
app = typer.Typer(
    name="strategy-report",
    help="A CLI for generating hybrid predictive and historical investment strategies.",
    add_completion=False,
)

# Single Console instance for consistent styling
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

    # --- 3. Orchestration (Prototype Logic) ---
    console.print("\n" + "="*50)
    console.print("[bold green]Query Received![/bold green]")
    console.print(f"  [bold]Company:[/bold] {selected_company_name}")
    console.print(f"  [bold]Ticker:[/bold] {selected_ticker}")
    console.print(f"  [bold]Horizon:[/bold] {selected_horizon}")
    console.print("="*50)

    # This is where we will delegate to the engine in the future
    if selected_horizon in constants.PREDICTIVE_HORIZONS:
        console.print("\n[italic yellow]Routing to Predictive Engine... (Not Implemented)[/italic yellow]")
        # Future call: report = engine.get_predictive_report(selected_ticker, selected_horizon)
        # Future call: display.show_predictive_report(report)
    else:
        console.print("\n[italic yellow]Routing to Historical Engine... (Not Implemented)[/italic yellow]")
        # Future call: report = engine.get_historical_report(selected_ticker, selected_horizon)
        # Future call: display.show_historical_report(report)

if __name__ == "__main__":
    app()