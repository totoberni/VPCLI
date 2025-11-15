import typer
import sys
from rich.console import Console, Group
from rich.prompt import Prompt
from rich.live import Live
from .core import constants
from . import engine, display

# --- Cross-platform single key press detection ---
try:
    # Windows
    import msvcrt
    def _get_key():
        key = msvcrt.getch()
        if key == b'\xe0': # Special key prefix
            key_code = msvcrt.getch()
            if key_code == b'K': return 'left'
            if key_code == b'M': return 'right'
        try:
            return key.decode('utf-8').lower()
        except UnicodeDecodeError:
            return None
except ImportError:
    # POSIX (Linux, macOS)
    import tty, termios
    def _get_key():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                next1 = sys.stdin.read(1)
                next2 = sys.stdin.read(1)
                if next1 == '[':
                    if next2 == 'D': return 'left'
                    if next2 == 'C': return 'right'
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch.lower()

app = typer.Typer(
    name="strategy-report",
    help="A CLI for generating hybrid predictive and historical investment strategies via an interactive session.",
    add_completion=False,
    rich_markup_mode="markdown"
)

console = Console()

def _select_company():
    console.print("\n[bold cyan]Welcome to the Financial Strategy Engine![/bold cyan]\n")
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
        - For historical reports, you can navigate pages using 'a' (left) and 'd' (right).

    After viewing a report, you will be prompted for your next action.
    """
    while True: # Outer loop for "Go Home"
        console.clear()
        selected_company_name, selected_ticker = _select_company()

        while True: # Inner loop for "New Horizon"
            console.clear()
            console.print(f"[bold]Company selected: [green]{selected_company_name}[/green][/bold]")
            selected_horizon = _select_horizon()
            show_volatility = Prompt.ask("Include volatility estimates in the report?", choices=["y", "n"], default="n") == "y"

            console.clear()

            try:
                if selected_horizon in constants.PREDICTIVE_HORIZONS:
                    console.print(f"[bold]Generating report for {selected_company_name} | {selected_horizon}...[/bold]\n")
                    report_data = engine.get_predictive_report(selected_ticker, selected_horizon, show_volatility)
                    display.show_predictive_report(report_data, selected_horizon, show_volatility)
                    Prompt.ask("\n[bold yellow]Press Enter to continue...[/bold yellow]")
                else:
                    pages, page_titles = engine.get_historical_pages(selected_ticker, selected_horizon)
                    if not pages:
                        console.print("[yellow]No historical signals found to display.[/yellow]")
                        Prompt.ask("\n[bold yellow]Press Enter to continue...[/bold yellow]")
                    else:
                        current_page_index = 0
                        
                        def get_page_renderable(index):
                            """Gets the renderable object for a given page index."""
                            return display.render_historical_page(
                                pages[index],
                                page_titles[index],
                                index + 1,
                                len(pages),
                                selected_horizon,
                                show_volatility
                            )

                        with Live(get_page_renderable(0), auto_refresh=False, vertical_overflow="scroll", transient=True) as live:

                            while True:
                                key = _get_key()
                
                                page_changed = False
                                if key in ['right', 'd'] and current_page_index < len(pages) - 1:
                                    current_page_index += 1
                                    page_changed = True

                                elif key in ['left', 'a'] and current_page_index > 0:
                                    current_page_index -= 1
                                    page_changed = True

                                elif key == 'q':
                                    break
                                
                                if page_changed:
                                    console.clear()
                                    live.update(get_page_renderable(current_page_index), refresh=True)
                                    

            except FileNotFoundError as e:
                console.print(f"[bold red]Error:[/bold red] {e}")
                Prompt.ask("\n[bold yellow]Press Enter to continue...[/bold yellow]")
            except Exception as e:
                console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")
                Prompt.ask("\n[bold yellow]Press Enter to continue...[/bold yellow]")

            # This is the definitive fix. Always clear the screen after ANY report has finished.
            console.clear()

            console.print("[bold]What's next?[/bold]")
            menu_options = { "1": "New Horizon (Same Company)", "2": "New Company / Go Home", "3": "Exit" }
            for key, value in menu_options.items(): console.print(f"  [yellow]{key}[/yellow]: {value}")
            
            next_action = Prompt.ask("Enter selection", choices=list(menu_options.keys()), default="3")
            
            if next_action == "1": 
                console.clear()
                continue
            elif next_action == "2": 
                console.clear()
                break 
            elif next_action == "3": 
                console.clear()
                console.print("\n[bold cyan]Exiting Strategy Engine. Goodbye![/bold cyan]")
                return

if __name__ == "__main__":
    app()