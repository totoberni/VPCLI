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

PAGE_SIZE = 20

PREDICTIVE_HORIZONS = ["1M", "3M", "6M", "12M", "18M", "24M", "36M"]

# REVERTED: Only include historical horizons that are supported by the data artifacts,
# while excluding the problematic 84M horizon.
HISTORICAL_HORIZONS = ["48M", "60M", "72M"]

ALL_HORIZONS = PREDICTIVE_HORIZONS + HISTORICAL_HORIZONS