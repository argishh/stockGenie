
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple
import json
from pathlib import Path

def get_company_aliases() -> List[Dict[str, str]]:
    """Returns a list of common company aliases and their ticker symbols"""
    # This is a starter list, you can expand it
    companies = [
        {"name": "Apple Inc.", "ticker": "AAPL", "aliases": ["Apple"]},
        {"name": "Microsoft Corporation", "ticker": "MSFT", "aliases": ["Microsoft"]},
        {"name": "Alphabet Inc.", "ticker": "GOOGL", "aliases": ["Google"]},
        {"name": "Amazon.com Inc.", "ticker": "AMZN", "aliases": ["Amazon"]},
        {"name": "Meta Platforms Inc.", "ticker": "META", "aliases": ["Facebook"]},
        {"name": "Tesla Inc.", "ticker": "TSLA", "aliases": ["Tesla"]},
        {"name": "Netflix Inc.", "ticker": "NFLX", "aliases": ["Netflix"]},
        {"name": "NVIDIA Corporation", "ticker": "NVDA", "aliases": ["NVIDIA"]},
        {"name": "PayPal Holdings Inc.", "ticker": "PYPL", "aliases": ["PayPal"]},
        {"name": "Adobe Inc.", "ticker": "ADBE", "aliases": ["Adobe"]},
        {"name": "Intel Corporation", "ticker": "INTC", "aliases": ["Intel"]},
        {"name": "Advanced Micro Devices Inc.", "ticker": "AMD", "aliases": ["AMD"]},
        {"name": "Salesforce.com Inc.", "ticker": "CRM", "aliases": ["Salesforce"]},
        {"name": "Cisco Systems Inc.", "ticker": "CSCO", "aliases": ["Cisco"]},
        {"name": "Oracle Corporation", "ticker": "ORCL", "aliases": ["Oracle"]},
        {"name": "IBM Corporation", "ticker": "IBM", "aliases": ["IBM"]},
        {"name": "Uber Technologies Inc.", "ticker": "UBER", "aliases": ["Uber"]},
        {"name": "Spotify Technology S.A.", "ticker": "SPOT", "aliases": ["Spotify"]},
        {"name": "Shopify Inc.", "ticker": "SHOP", "aliases": ["Shopify"]},
    ]
    return companies

def find_ticker(query: str) -> Tuple[str, str]:
    """Find ticker symbol from company name or alias"""
    query = query.lower()
    for company in get_company_aliases():
        if query in [x.lower() for x in company["aliases"]]:
            return company["ticker"], company["name"]
    return query.upper(), query.upper()  # Return as-is if not found