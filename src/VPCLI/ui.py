import sys
from rich.console import Console
from rich.prompt import Prompt
from .core import constants

# --- Cross-platform single key press detection ---
try:
    # Windows
    import msvcrt

    def get_key():
        key = msvcrt.getch()
        if key == b"\xe0":  # Special key prefix for arrow keys
            key_code = msvcrt.getch()
            if key_code == b"K":
                return "left"
            if key_code == b"M":
                return "right"
        try:
            return key.decode("utf-8").lower()
        except UnicodeDecodeError:
            return None


except ImportError:
    # POSIX (Linux, macOS)
    import tty, termios

    def get_key():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
            if ch == "\x1b":  # Arrow key escape sequence
                next1 = sys.stdin.read(1)
                next2 = sys.stdin.read(1)
                if next1 == "[":
                    if next2 == "D":
                        return "left"
                    if next2 == "C":
                        return "right"
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch.lower()


console = Console()


def select_company():
    """Prompts the user to select a company."""
    console.print("\n[bold cyan]Welcome to the Financial Strategy Engine![/bold cyan]\n")
    company_prompt_text = "\n".join(
        [f"  [bold yellow]{key}[/bold yellow]: {name}" for key, name in constants.COMPANIES.items()]
    )
    console.print("[bold]First, select a company to investigate:[/bold]")
    console.print(company_prompt_text)
    company_idx = Prompt.ask(
        "Enter selection", choices=list(constants.COMPANIES.keys()), default="2"
    )
    return constants.COMPANIES[company_idx], constants.COMPANY_TICKERS[company_idx]


def select_horizon():
    """Prompts the user to select an investment horizon."""
    console.print("\n[bold]Next, select an investment horizon:[/bold]")
    console.print(
        "\n[bold magenta]--- Predictive Recommendations (1M-36M) ---[/bold magenta]"
    )
    console.print("  " + " | ".join(constants.PREDICTIVE_HORIZONS))
    console.print(
        "\n[bold magenta]--- Historical Certainty Analysis (48M-84M) ---[/bold magenta]"
    )
    console.print("  " + " | ".join(constants.HISTORICAL_HORIZONS))
    return Prompt.ask(
        "\nEnter horizon", choices=constants.ALL_HORIZONS, default="12M"
    )


def get_next_action():
    """Prompts the user for the next action after a report is displayed."""
    console.print("\n[bold]What's next?[/bold]")
    menu_options = {
        "1": "New Horizon (Same Company)",
        "2": "New Company / Go Home",
        "3": "Exit",
    }
    for key, value in menu_options.items():
        console.print(f"  [yellow]{key}[/yellow]: {value}")
    return Prompt.ask("Enter selection", choices=list(menu_options.keys()), default="3")