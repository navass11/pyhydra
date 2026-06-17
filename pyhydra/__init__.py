from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pyhydra")
except PackageNotFoundError:
    __version__ = "unknown"

# Azure Container Apps (Consumption, no VNet) resolves hostnames to IPv6 but
# has no IPv6 routing, causing ENETUNREACH on every outbound HTTP request.
# Force IPv4 for all urllib3/requests calls made by pyhydra.
try:
    import socket
    import urllib3.util.connection as _urllib3_conn
    _urllib3_conn.allowed_gai_family = lambda: socket.AF_INET
except Exception:
    pass
