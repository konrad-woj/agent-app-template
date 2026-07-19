"""SSRF protection for MCP tool URLs.

Provides two validation layers:
  1. _validate_url  — scheme allow-list + literal private-IP block (fast, no DNS).
  2. validate_url_and_host — calls _validate_url then resolves the hostname via DNS
     and re-validates every returned IP.  Prevents DNS rebinding attacks where an
     attacker-controlled hostname initially resolves to a public IP but later
     resolves to an internal address.

Usage in MCP tools:
    from app.mcp.ssrf import validate_url_and_host
    await validate_url_and_host(url)  # raises ValueError if blocked
"""

import asyncio
import ipaddress
import socket
from urllib.parse import urlparse

_ALLOWED_SCHEMES = {"http", "https"}


def _validate_url(url: str) -> None:
    """Reject non-HTTP(S) schemes and private/loopback/link-local hosts.

    Prevents SSRF: the LLM can be prompt-injected via search results to call
    fetch_url with internal addresses (e.g. 169.254.169.254 AWS metadata).
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"Disallowed URL scheme: {parsed.scheme!r}")
    host = parsed.hostname or ""
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        # Not an IP literal — check private hostname patterns
        if host == "localhost" or host.endswith(".local") or host.endswith(".internal"):
            raise ValueError(f"Blocked private hostname: {host}")  # noqa: B904
        return
    if (
        not addr.is_global
        or addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
    ):
        raise ValueError(f"Blocked non-public IP: {host}")


async def validate_url_and_host(url: str) -> str:
    """Validate URL and resolve hostname to block DNS rebinding.

    Applies _validate_url first, then resolves the hostname via
    socket.getaddrinfo (off the event loop, via asyncio.to_thread — a hung
    resolver must not stall the MCP server's event loop) and re-validates
    every returned IP address against the same private/loopback/link-local
    block list.

    Note: this is a defense-in-depth measure, not a complete TOCTOU fix —
    the underlying HTTP client re-resolves DNS at connect time.

    Args:
        url: The URL to validate.

    Returns:
        The original URL if all checks pass.

    Raises:
        ValueError: If the URL fails any validation check.
    """
    _validate_url(url)
    parsed = urlparse(url)
    host = parsed.hostname or ""
    # Literal IPs were already validated by _validate_url — skip re-resolution.
    try:
        ipaddress.ip_address(host)
        return url
    except ValueError:
        pass
    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, None)
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve hostname: {host!r}") from exc
    for info in infos:
        addr_str = info[4][0]
        try:
            addr = ipaddress.ip_address(addr_str)
        except ValueError:
            continue
        if (
            not addr.is_global
            or addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_multicast
        ):
            raise ValueError(f"Hostname {host!r} resolves to blocked IP: {addr_str}")
    return url
