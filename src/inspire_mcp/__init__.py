"""INSPIRE-HEP MCP server."""

__version__ = "0.2.0"

from .client import Author, InspireClient, InspireError, Paper  # noqa: E402
from .server import build_server  # noqa: E402

__all__ = ["Author", "InspireClient", "InspireError", "Paper", "build_server", "__version__"]
