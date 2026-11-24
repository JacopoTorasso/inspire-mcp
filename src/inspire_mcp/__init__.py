"""INSPIRE-HEP MCP server."""

from .client import Author, InspireClient, InspireError, Paper
from .server import build_server

__all__ = ["InspireClient", "InspireError", "Paper", "Author", "build_server"]
