
"""Generate a clearly labelled synthetic dataset for local development."""

import pandas as pd


def load_sample_offers() -> pd.DataFrame:
    """Return illustrative reward offers; these are not real Freecash offers."""
    offers = [
        {
            "offer_id": "SYN-001",
            "offer_name": "Example: Reach Level 10",
            "category": "Mobile Gaming",
            "advertised_reward_usd": 25.00,
            "estimated_days": 7,
            "difficulty": "Medium",
            "conditions": "Reach level 10 within 14 days of installation.",
            "completion_rate_pct": 72,
            "sample_status": "Synthetic",
        },
        {
            "offer_id": "SYN-002",
            "offer_name": "Example: Complete 5 Surveys",
            "category": "Surveys",
            "advertised_reward_usd": 3.50,
            "estimated_days": 1,
            "difficulty": "Easy",
            "conditions": "Complete five eligible surveys; screening may apply.",
            "completion_rate_pct": 85,
            "sample_status": "Synthetic",
        },
        {
            "offer_id": "SYN-003",
            "offer_name": "Example: Reach Level 30",
            "category": "Mobile Gaming",
            "advertised_reward_usd": 80.00,
            "estimated_days": 21,
            "difficulty": "Hard",
            "conditions": "Reach level 30 within 30 days; eligibility requirements apply.",
            "completion_rate_pct": 38,
            "sample_status": "Synthetic",
        },
        {
            "offer_id": "SYN-004",
            "offer_name": "Example: Register for a Trial",
            "category": "Subscriptions",
            "advertised_reward_usd": 12.00,
            "estimated_days": 2,
            "difficulty": "Medium",
            "conditions": "Register for an eligible trial and meet its terms.",
            "completion_rate_pct": 58,
            "sample_status": "Synthetic",
        },
        {
            "offer_id": "SYN-005",
            "offer_name": "Example: Reach Level 5",
            "category": "Mobile Gaming",
            "advertised_reward_usd": 8.00,
            "estimated_days": 3,
            "difficulty": "Easy",
            "conditions": "Reach level 5 within 7 days of installation.",
            "completion_rate_pct": 78,
            "sample_status": "Synthetic",
        },
        {
            "offer_id": "SYN-006",
            "offer_name": "Example: Complete a Purchase",
            "category": "Shopping",
            "advertised_reward_usd": 15.00,
            "estimated_days": 4,
            "difficulty": "Hard",
            "conditions": "Make an eligible purchase and satisfy the offer terms.",
            "completion_rate_pct": 45,
            "sample_status": "Synthetic",
        },
    ]

    return pd.DataFrame(offers)
