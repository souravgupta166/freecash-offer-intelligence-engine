
"""Registry of candidate sources and their collection permissions."""

SOURCES = {
    "freecash": {
        "name": "Freecash",
        "url": "https://freecash.com/",
        "data_type": "Offer descriptions and reward conditions",
        "access_method": "Written authorization required before automated collection",
        "approved_for_automated_collection": False,
        "notes": (
            "Current Terms of Service restrict automated access and copying. "
            "Do not run a scraper without explicit authorization."
        ),
    },
    "trustpilot": {
        "name": "Trustpilot",
        "url": "https://www.trustpilot.com/",
        "data_type": "Public business reviews",
        "access_method": "Authorized API or explicitly permitted data access",
        "approved_for_automated_collection": False,
        "notes": (
            "Terms prohibit scraping and data mining without express permission. "
            "Official API access must be arranged with Trustpilot."
        ),
    },
}


def get_source_status() -> list[dict]:
    """Return source metadata without accessing either website."""
    return [
        {"source_id": source_id, **details}
        for source_id, details in SOURCES.items()
    ]
