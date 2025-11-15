import typer
from rich.console import Console
from rich.prompt import Prompt
from .core import constants
from . import engine, display

app = typer.Typer(
    name="strategy-report",
    help="A CLI for generating hybrid predictive and historical investment strategies via an interactive session.",
    add_completion=False,
    rich_markup_mode="markdown"
)

console = Console()

def _select_company():
    console.clear()
    company_prompt_text = "\n".join([f"  [bold yellow]{key}[/bold yellow]: {name}" for key, name in constants.COMPANIES.items()])
    console.print("[bold]First, select a company to investigate:[/bold]")
    console.print(company_prompt_text)
    company_idx = Prompt.ask("Enter selection", choices=list(constants.COMPANIES.keys()), default="2")
    return constants.COMPANIES[company_idx], constants.COMPANY_TICKERS[company_idx]

def _select_horizon():
    console.print("\n[bold]Next, select an investment horizon:[/bold]")
    console.print("\n[bold magenta]--- Predictive Recommendations (1M-36M) ---[/bold magenta]")
    console.print("  " + " | ".join(constants.PREDICTIVE_HORIZONS))
    console.print("\n[bold magenta]--- Historical Certainty Analysis (48M-84M) ---[/bold magenta]")
    console.print("  " + " | ".join(constants.HISTORICAL_HORIZONS))
    return Prompt.ask("\nEnter horizon", choices=constants.ALL_HORIZONS, default="12M")

@app.command(help="Start the main interactive session to generate strategy reports.")
def run():
    """
    Starts the main interactive session.

    This command will guide you through a series of prompts to generate a report:

    1.  **Select a Company**: Choose from AAPL, NVDA, or GOOGL.
    2.  **Select an Investment Horizon**: Choose a predictive (1M-36M) or historical (48M-84M) horizon.
    3.  **View Report**: The generated report will be displayed.
        - For historical reports, you can navigate through monthly pages using the [P]revious, [N]ext, and [Q]uit commands.

    After viewing a report, you will be prompted for your next action: **New Horizon**, **New Company**, or **Exit**.
    """
    console.print("\n[bold cyan]Welcome to the Financial Strategy Engine![/bold cyan]\n")
    
    selected_company_name, selected_ticker = _select_company()

    while True:
        console.clear()
        console.print(f"[bold]Company selected: [green]{selected_company_name}[/green][/bold]")
        selected_horizon = _select_horizon()

        console.print("\n" + "="*50)
        console.print(f"[bold]Generating report for {selected_company_name} | {selected_horizon}...[/bold]")
        console.print("="*50 + "\n")
        
        show_volatility = Prompt.ask("Include volatility estimates in the report?", choices=["y", "n"], default="n") == "y"

        display_status = None
        try:
            console.clear()
            if selected_horizon in constants.PREDICTIVE_HORIZONS:
                console.print("[italic yellow]Running predictive simulation...[/italic yellow]\n")
                report_data = engine.get_predictive_report(selected_ticker, selected_horizon, show_volatility)
                display.show_predictive_report(report_data, selected_horizon, show_volatility)
            else:
                console.print("[italic yellow]Fetching historical analysis...[/italic yellow]\n")
                report_data = engine.get_historical_report(selected_ticker, selected_horizon)
                display_status = display.show_historical_report(report_data, selected_horizon, show_volatility)
        except FileNotFoundError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            console.print("[yellow]Please ensure all required .pkl artifacts are present in the 'assets' directory.[/yellow]")
        except Exception as e:
            console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")

        # If the user chose to quit the pager, exit the whole app
        if display_status == 'exit':
             console.print("\n[bold cyan]Exiting Strategy Engine. Goodbye![/bold cyan]")
             break

        console.print("\n" + "="*50)
        next_action = Prompt.ask("What's next?", choices=["New Horizon", "New Company", "Exit"], default="Exit")
        
        if next_action == "Exit":
            console.print("\n[bold cyan]Exiting Strategy Engine. Goodbye![/bold cyan]")
            break
        elif next_action == "New Company":
            selected_company_name, selected_ticker = _select_company()
        # If "New Horizon", the loop continues with the same company

if __name__ == "__main__":
    app()