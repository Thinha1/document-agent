"""Build the retrieval-focused model tool surface.

The built-in interface is deliberately small: self-hosted search. Operators
can add document stores or other retrieval adapters through MCP.
"""

from __future__ import annotations

from tini.config import Settings
from tini.tools import documents, search
from tini.tools.registry import ToolRegistry


def build_registry(settings: Settings, document_index=None) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(search.make_tool())
    if document_index is not None:
        registry.register(documents.make_tool(document_index, settings.document_workspace))

    # Internal document systems commonly expose an MCP server. The bridge is
    # optional and only starts when the operator created .tini/mcp.json.
    mcp_config = settings.home / "mcp.json"
    if mcp_config.exists():
        try:
            from tini.tools.mcp_client import MCPBridge

            bridge = MCPBridge(mcp_config)
            for tool in bridge.start():
                registry.register(tool)
            registry.mcp_bridge = bridge
        except ImportError:
            print("mcp.json found but the 'mcp' package is missing — pip install 'tini-agent[mcp]'")

    return registry
