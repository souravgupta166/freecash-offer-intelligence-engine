
import pytest

from src.processing.label_reviews import assign_proxy_sentiment


@pytest.mark.parametrize(
    ("rating", "expected"),
    [
        (0, None),
        (1, "negative"),
        (2, "negative"),
        (3, "neutral"),
        (4, "positive"),
        (5, "positive"),
    ],
)
def test_rating_to_proxy_sentiment(rating, expected):
    assert assign_proxy_sentiment(rating) == expected


def test_missing_rating_returns_none():
    assert assign_proxy_sentiment(float("nan")) is None


@pytest.mark.parametrize("rating", [-1, 6, 2.5])
def test_invalid_rating_raises_error(rating):
    with pytest.raises(ValueError):
        assign_proxy_sentiment(rating)
