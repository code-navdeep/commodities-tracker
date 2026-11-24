"""Shared HTTP client for all sources.

CME's public endpoints and delivery PDFs sit behind Akamai bot detection that
instantly 403s plain HTTP clients; curl_cffi with Chrome TLS impersonation
passes cleanly. This module is the only place that detail lives.

Failures retry once on a *fresh* session: long fetch runs can have the server
kill the HTTP/2 connection mid-stream, after which every request on the same
session fails the same way — so a failed request never reuses its session.
"""

import logging
import time

from curl_cffi import requests as cffi_requests

logger = logging.getLogger(__name__)

TIMEOUT = 45
RETRY_WAIT = 3

_session = None


def _get_session():
    global _session
    if _session is None:
        _session = cffi_requests.Session(impersonate='chrome')
    return _session


def _reset_session():
    global _session
    if _session is not None:
        try:
            _session.close()
        except Exception:
            pass
        _session = None


def _get(url: str, params: dict = None):
    last_error = None
    for attempt in (1, 2):
        try:
            response = _get_session().get(url, params=params, timeout=TIMEOUT)
            if response.status_code == 200:
                return response
            last_error = f"HTTP {response.status_code}"
        except Exception as e:
            last_error = str(e)
        _reset_session()
        if attempt == 1:
            logger.warning(f"Request failed ({last_error}), retrying: {url}")
            time.sleep(RETRY_WAIT)
    raise RuntimeError(f"GET failed after retry: {url} — {last_error}")


def get_json(url: str, params: dict = None):
    return _get(url, params).json()


def get_bytes(url: str, params: dict = None) -> bytes:
    return _get(url, params).content


def get_text(url: str, params: dict = None) -> str:
    return _get(url, params).text
