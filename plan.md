# Financial Strategist CLI: Detailed Implementation Plan

**Objective:** This document provides a detailed, step-by-step technical guide for scaffolding, developing, and deploying the `financial_strategist_cli` tool. The plan emphasizes a modular, extensible, and cross-platform architecture from the outset.

---

## Phase 1: Project Scaffolding & Environment Setup

This phase establishes the foundational structure and isolated environment for the project.

### **Step 1.1: Project Directory Structure**

A well-defined structure is critical for maintainability and scalability. The following layout separates concerns (code, assets, tests) and aligns with Python packaging best practices.

**Action:** Create the following directory and file structure.

```
VPCLI/
│
├── .venv/                     # (Local Development) Virtual environment
├── assets/                    # For storing data artifacts
│   └── .gitkeep               # Placeholder
├── src/                       # Main source code
│   ├── __init__.py            
│   ├── main.py                # CLI entry point (Typer app)
│   ├── engine.py              # Core business logic
│   ├── display.py             # Presentation logic
│   └── core/                  
│       ├── __init__.py
│       └── constants.py       # Shared constants
├── tests/                     # For unit and integration tests
│   └── __init__.py
├── .gitignore                 # Specifies files for Git to ignore
├── Dockerfile                 # (Deployment) Recipe for building the application container
├── pyproject.toml             # Project metadata, dependencies, and script entry point
└── README.md                  # Project overview and instructions
```

**Architectural Decisions:**
- **`src` Layout:** Using a `src` directory is a modern standard that prevents common Python import problems and ensures the installed package has the same structure as the development source.
- **Modularity:** `main.py`, `engine.py`, and `display.py` are separated from day one. `main.py` handles user interaction, `engine.py` runs the analysis, and `display.py` formats the output. This separation is key for testing and future expansion.
- **`assets`:** This directory is designated for the "static" data artifacts (`.pkl`, `.html`) that the CLI will consume. This makes it clear what the inputs to the application are.
- **`tests`:** A dedicated `tests` directory is included to facilitate a test-driven development (TDD) approach later on.


### **Step 1.2: Local Development Environment**

A local virtual environment (`.venv`) is used by developers for a fast, iterative coding workflow.

1.  **Creation:** From the project root, run:
    `python -m venv .venv`
2.  **Activation:**
    -   **Windows:** `.\.venv\Scripts\Activate.ps1`
    -   **macOS/Linux:** `source .venv/bin/activate`

---

## Phase 2: Dependency Management

We use `pyproject.toml` to define the Python-level dependencies for both local development and the final Docker container.

### **Step 2.1: `pyproject.toml`**

```toml
[project]
name = "financial_strategist_cli"
version = "0.1.0"
description = "A CLI tool for providing hybrid predictive and historical investment strategies."
dependencies = [
    "typer[all] >= 0.9.0",
    "rich >= 13.0.0",
    "pandas == 2.3.3",
    "xgboost == 3.1.1",
    "scikit-learn == 1.7.2",
]

[project.scripts]
strategy-report = "src.main:app"

[tool.setuptools.packages.find]
where = ["src"]
```

### **Step 2.2: Install Local Dependencies**

With the virtual environment activated, run this command to install the project and its dependencies for local development.

```bash
pip install -e .
```

---

## Phase 3: Application Core & Interactive Prototype

This phase focuses on creating a functional, interactive shell. We will create placeholders for the engine and display logic to prepare for future implementation.

### **Step 3.1: Create `src/core/constants.py`**

Centralize all static lists and configuration. This makes the code cleaner and easier to update.

```python
# src/core/constants.py

COMPANIES = {
    "1": "Apple (AAPL)",
    "2": "NVIDIA (NVDA)",
    "3": "Google (GOOGL)"
}

# Extracts "NVDA" from "NVIDIA (NVDA)"
COMPANY_TICKERS = {
    key: name.split('(')[-1].replace(')', '')
    for key, name in COMPANIES.items()
}

PREDICTIVE_HORIZONS = ["1M", "3M", "6M", "12M", "18M", "24M", "36M"]
HISTORICAL_HORIZONS = ["48M", "60M", "72M", "84M"]
ALL_HORIZONS = PREDICTIVE_HORIZONS + HISTORICAL_HORIZONS
```

### **Step 3.2: Create `src/main.py` (Interactive Entry Point)**

This script handles user interaction and orchestrates calls to the engine and display modules. For now, it will simply confirm the user's selections.

```python
# src/main.py

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
```

### **Step 3.3: Create Placeholders for `engine.py` and `display.py`**

Create these files with empty functions to make the architecture explicit.

```python
# src/engine.py

def get_predictive_report(ticker: str, horizon: str):
    """
    Loads predictive models and runs simulations.
    (This will be implemented in Phase 4)
    """
    print("Engine: Predictive report generation not yet implemented.")
    return None

def get_historical_report(ticker: str, horizon: str):
    """
    Loads pre-computed historical analysis reports.
    (This will be implemented in Phase 4)
    """
    print("Engine: Historical report generation not yet implemented.")
    return None
```

```python
# src/display.py

def show_predictive_report(report_data):
    """
    Formats and displays the predictive rulebook using rich.
    (This will be implemented in Phase 4)
    """
    print("Display: Predictive report display not yet implemented.")

def show_historical_report(report_data):
    """
    Formats and displays the historical certainty report using rich.
    (This will be implemented in Phase 4)
    """
    print("Display: Historical report display not yet implemented.")
```
---

## Phase 4: Verification & Next Steps

This final step ensures our prototype is working as expected.

1.  **Run the application** from the project root directory:
    ```bash
    strategy-report
    ```
2.  **Interact with the prompts.** The tool should correctly ask for a company and horizon, confirm your choices, and print the "Not Implemented" message for the corresponding engine path.
3.  **Check the help message,** which is automatically generated by Typer:
    ```bash
    strategy-report --help
    ```

---

## Phase 5: Packaging & Deployment Strategy (Docker)

This phase details how to package the finished application into a self-contained, portable Docker container for distribution. This is the **deployment** workflow, performed after local development and testing are complete.

### **Step 5.1: Create the `Dockerfile`**

Create a file named `Dockerfile` in the project root. This file is the blueprint for our application environment.

```dockerfile
# Stage 1: Base Image
# Start from an official NVIDIA image that includes the required CUDA toolkit.
# This ensures GPU compatibility for the XGBoost models.
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Stage 2: System & Python Setup
# Install Python 3.12 and pip.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    python3-pip \
    python3.12-venv && \
    rm -rf /var/lib/apt/lists/*

# Make python3.12 the default python command
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Stage 3: Application Setup
# Set the working directory inside the container.
WORKDIR /app

# Copy only the necessary files for dependency installation first.
# This leverages Docker's layer caching.
COPY pyproject.toml .

# Install Python dependencies. The `pip install .` command reads the
# pyproject.toml file and installs the application along with its dependencies.
RUN python3 -m pip install .

# Copy the rest of the application source code and static assets.
COPY src/ ./src
COPY assets/ ./assets

# Stage 4: Execution
# Define the entry point for the container. This makes the container
# act like an executable for our 'strategy-report' command.
ENTRYPOINT ["strategy-report"]

# Define the default command to run when the container starts.
# This can be overridden from the `docker run` command line.
CMD ["run"]
```

### **Step 5.2: Build the Docker Image**

From the project root, run the `docker build` command. This will execute the steps in the `Dockerfile` and create a portable image containing your application.

```bash
# The -t flag "tags" the image with a name (e.g., financial-strategist-cli)
docker build -t financial-strategist-cli .
```

### **Step 5.3: Run the Container**

Once built, anyone with Docker can run your application with a single command, without worrying about installing Python, CUDA, or any other dependencies.

```bash
# --rm: Automatically remove the container when it exits.
# -it: Run in interactive mode to use the CLI prompts.
# --gpus all: Provide the container with access to the host's NVIDIA GPUs.
docker run --rm -it --gpus all financial-strategist-cli
```
```