import sys
import os
import typer
from rich.console import Console
from rich.prompt import Prompt

from .core import constants
from . import engine, display, ui  # Import the new ui module

app = typer.Typer(
    name="strategy-report",
    help="A CLI for generating hybrid predictive and historical investment strategies.",
    add_completion=False,
    rich_markup_mode="markdown",
)

console = Console()


@app.command(help="Start the main interactive session to generate strategy reports.")
def run():
    """
    Starts the main interactive session.
    This command will guide you through a series of prompts to generate a report.
    """
    if sys.platform == "win32":
        os.system("chcp 65001 > nul")

    while True:  # Outer loop for "Go Home" / "New Company"
        console.clear()
        selected_company_name, selected_ticker = ui.select_company()

        while True:  # Inner loop for "New Horizon"
            console.clear()
            console.print(f"[bold]Company selected: [green]{selected_company_name}[/green][/bold]")
            selected_horizon = ui.select_horizon()
            show_volatility = (
                Prompt.ask(
                    "Include volatility estimates in the report?",
                    choices=["y", "n"],
                    default="n",
                )
                == "y"
            )

            try:
                if selected_horizon in constants.PREDICTIVE_HORIZONS:
                    console.clear()
                    console.print(
                        f"[bold]Generating report for {selected_company_name} | {selected_horizon}...[/bold]\n"
                    )
                    report_data = engine.get_predictive_report(
                        selected_ticker, selected_horizon, show_volatility
                    )
                    display.show_predictive_report(
                        report_data, selected_horizon, show_volatility
                    )
                    Prompt.ask("\n[bold yellow]Press Enter to continue...[/bold yellow]")
                else:
                    # --- Historical path with NEW time-aware logic ---
                    console.print("[italic yellow]Processing time-aware historical analysis...[/italic yellow]")
                    monthly_summaries = engine.get_monthly_signal_summary(
                        selected_ticker, selected_horizon
                    )
                    
                    if not monthly_summaries:
                        console.clear()
                        console.print("[yellow]No historical signals found to display.[/yellow]")
                        Prompt.ask("\n[bold yellow]Press Enter to continue...[/bold yellow]")
                    else:
                        current_page_index = 0
                        while True:
                            console.clear()
                            # Get the month and its corresponding dataframe
                            month, month_df = monthly_summaries[current_page_index]
                            
                            # Updated call with new arguments
                            display.render_historical_page(
                                month_df,
                                month,
                                current_page_index + 1,
                                len(monthly_summaries),
                                selected_horizon,
                                show_volatility,
                            )

                            key = ui.get_key()
                            if key in ["right", "d"] and current_page_index < len(monthly_summaries) - 1:
                                current_page_index += 1
                            elif key in ["left", "a"] and current_page_index > 0:
                                current_page_index -= 1
                            elif key in ["q", "Q"]:
                                break

            except FileNotFoundError as e:
                console.clear()
                console.print(f"[bold red]Error:[/bold red] {e}")
                Prompt.ask("\n[bold yellow]Press Enter to continue...[/bold yellow]")
            except Exception as e:
                console.clear()
                console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")
                Prompt.ask("\n[bold yellow]Press Enter to continue...[/bold yellow]")

            console.clear()
            next_action = ui.get_next_action()

            if next_action == "1":  # New Horizon
                continue
            elif next_action == "2":  # New Company
                break
            elif next_action == "3":  # Exit
                console.clear()
                console.print("\n[bold cyan]Exiting Strategy Engine. Goodbye![/bold cyan]")
                return


if __name__ == "__main__":
    app()