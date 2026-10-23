
"""Check website reachability without collecting page content.

Note:
    A successful HTTP response does not grant permission to scrape,
    store, or reuse website content.
"""

from datetime import datetime, timezone
from urllib.parse import urlparse

import requests


USER_AGENT = (
    "FreecashOfferIntelligenceResearch/0.1 "
    "(contact: project-maintainer)"
)


def check_website_access(url: str, timeout: int = 10) -> dict:
    """Check the HTTP reachability of an HTTPS URL.

    Args:
        url: The HTTPS URL to check.
        timeout: Maximum request duration in seconds.

    Returns:
        A dictionary containing the check timestamp, HTTP status,
        content type, reachability and any network error.

    Raises:
        ValueError: If the URL is invalid or does not use HTTPS.
    """
    if not isinstance(url, str):
        raise ValueError("Only valid HTTPS URLs are supported.")

    parsed = urlparse(url)

    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("Only valid HTTPS URLs are supported.")

    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise ValueError("Timeout must be a positive number.")

    result = {
        "url": url,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "reachable": False,
        "http_status": None,
        "content_type": None,
        "error": None,
        "permission_to_collect": False,
    }

    try:
        # HEAD requests headers only; no page body is intentionally downloaded.
        response = requests.head(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            allow_redirects=True,
        )

        try:
            result["http_status"] = response.status_code
            result["content_type"] = response.headers.get("Content-Type")
            result["reachable"] = response.status_code < 500
        finally:
            response.close()

    except requests.RequestException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result
