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