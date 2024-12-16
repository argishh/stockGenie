
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple
import json
from pathlib import Path

def get_company_aliases() -> List[Dict[str, str]]:
    """Returns a list of common company aliases and their ticker symbols"""
    # This is a starter list, you can expand it
    companies = [
        {"name": "Apple Inc.", "ticker": "AAPL", "aliases": ["Apple", "AAPL"]},
        {"name": "Microsoft Corporation", "ticker": "MSFT", "aliases": ["Microsoft", "MSFT"]},
        {"name": "Alphabet Inc.", "ticker": "GOOGL", "aliases": ["Google", "GOOGL", "Alphabet"]},
        {"name": "Amazon.com Inc.", "ticker": "AMZN", "aliases": ["Amazon", "AMZN"]},
        {"name": "Meta Platforms Inc.", "ticker": "META", "aliases": ["Facebook", "Meta", "FB"]},
        {"name": "Tesla Inc.", "ticker": "TSLA", "aliases": ["Tesla", "TSLA"]},
        {"name": "Netflix Inc.", "ticker": "NFLX", "aliases": ["Netflix", "NFLX"]},
        {"name": "NVIDIA Corporation", "ticker": "NVDA", "aliases": ["NVIDIA", "NVDA"]},
        {"name": "PayPal Holdings Inc.", "ticker": "PYPL", "aliases": ["PayPal", "PYPL"]},
        {"name": "Adobe Inc.", "ticker": "ADBE", "aliases": ["Adobe", "ADBE"]},
        {"name": "Intel Corporation", "ticker": "INTC", "aliases": ["Intel", "INTC"]},
        {"name": "Advanced Micro Devices Inc.", "ticker": "AMD", "aliases": ["AMD", "Advanced Micro Devices"]},
        {"name": "Salesforce.com Inc.", "ticker": "CRM", "aliases": ["Salesforce", "CRM"]},
        {"name": "Cisco Systems Inc.", "ticker": "CSCO", "aliases": ["Cisco", "CSCO"]},
        {"name": "Oracle Corporation", "ticker": "ORCL", "aliases": ["Oracle", "ORCL"]},
        {"name": "IBM Corporation", "ticker": "IBM", "aliases": ["IBM"]},
        {"name": "Uber Technologies Inc.", "ticker": "UBER", "aliases": ["Uber", "UBER"]},
        {"name": "Spotify Technology S.A.", "ticker": "SPOT", "aliases": ["Spotify", "SPOT"]},
        {"name": "Shopify Inc.", "ticker": "SHOP", "aliases": ["Shopify", "SHOP"]},
    ]
    return companies

def find_ticker(query: str) -> Tuple[str, str]:
    """Find ticker symbol from company name or alias"""
    query = query.lower()
    for company in get_company_aliases():
        if query in [x.lower() for x in company["aliases"]]:
            return company["ticker"], company["name"]
    return query.upper(), query.upper()  # Return as-is if not found