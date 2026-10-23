
"""Tests for the website access checker."""

from unittest.mock import Mock, patch

import pytest
import requests

from src.collectors.access_checker import check_website_access


def test_rejects_non_https_url():
    """HTTP URLs must be rejected."""
    with pytest.raises(ValueError, match="Only valid HTTPS URLs"):
        check_website_access("http://example.com")


def test_rejects_invalid_url():
    """Malformed URLs must be rejected."""
    with pytest.raises(ValueError, match="Only valid HTTPS URLs"):
        check_website_access("not-a-valid-url")


def test_rejects_url_without_hostname():
    """HTTPS URLs without a hostname must be rejected."""
    with pytest.raises(ValueError, match="Only valid HTTPS URLs"):
        check_website_access("https:///path")


def test_rejects_url_with_credentials():
    """URLs containing embedded credentials must be rejected."""
    with pytest.raises(ValueError, match="Only valid HTTPS URLs"):
        check_website_access("https://user:password@example.com")


@pytest.mark.parametrize("timeout", [0, -1, True, "5", None])
def test_rejects_invalid_timeout(timeout):
    """Timeout must be a positive number."""
    with pytest.raises(ValueError, match="Timeout must be a positive number"):
        check_website_access("https://example.com", timeout=timeout)


@patch("src.collectors.access_checker.requests.head")
def test_successful_response(mock_head):
    """A successful response returns HTTP metadata."""
    response = Mock()
    response.status_code = 200
    response.headers = {"Content-Type": "text/html"}
    mock_head.return_value = response

    result = check_website_access("https://example.com")

    assert result["reachable"] is True
    assert result["http_status"] == 200
    assert result["content_type"] == "text/html"
    assert result["error"] is None
    assert result["permission_to_collect"] is False

    mock_head.assert_called_once()
    response.close.assert_called_once()


@patch("src.collectors.access_checker.requests.head")
def test_server_error_is_not_reachable(mock_head):
    """HTTP 5xx responses are marked as not reachable."""
    response = Mock()
    response.status_code = 503
    response.headers = {}
    mock_head.return_value = response

    result = check_website_access("https://example.com")

    assert result["reachable"] is False
    assert result["http_status"] == 503
    response.close.assert_called_once()


@patch("src.collectors.access_checker.requests.head")
def test_network_error_is_captured(mock_head):
    """Network errors are returned in the result rather than raised."""
    mock_head.side_effect = requests.ConnectionError("Network unavailable")

    result = check_website_access("https://example.com", timeout=3)

    assert result["reachable"] is False
    assert result["http_status"] is None
    assert "ConnectionError" in result["error"]
    assert result["permission_to_collect"] is False


@patch("src.collectors.access_checker.requests.head")
def test_request_uses_head_and_timeout(mock_head):
    """The checker uses a HEAD request with the configured timeout."""
    response = Mock()
    response.status_code = 200
    response.headers = {}
    mock_head.return_value = response

    check_website_access("https://example.com", timeout=5)

    _, kwargs = mock_head.call_args
    assert kwargs["timeout"] == 5
    assert kwargs["allow_redirects"] is True
    assert "User-Agent" in kwargs["headers"]
