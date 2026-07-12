"""ToolRegistry - central registry for managing tools."""

from typing import Dict, List

from agent_framework.interfaces.base_tool import BaseTool


class ToolRegistry:
    """Registry for managing tool instances by name.

    Supports registering, unregistering, and retrieving tools.
    Tool names must be unique within the registry.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance.

        Args:
            tool: The tool to register. tool.name is used as the key.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered. "
                "Unregister it first to replace."
            )
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name.

        Args:
            name: The name of the tool to remove.

        Raises:
            KeyError: If no tool with the given name is registered.
        """
        if name not in self._tools:
            raise KeyError(f"No tool registered with name '{name}'.")
        del self._tools[name]

    def get(self, name: str) -> BaseTool:
        """Retrieve a registered tool by name.

        Args:
            name: The name of the tool to retrieve.

        Returns:
            The registered BaseTool instance.

        Raises:
            KeyError: If no tool with the given name is registered.
        """
        if name not in self._tools:
            raise KeyError(f"No tool registered with name '{name}'.")
        return self._tools[name]

    def list_tools(self) -> List[str]:
        """List all registered tool names.

        Returns:
            List of registered tool name strings.
        """
        return list(self._tools.keys())

    def to_dict(self) -> Dict[str, BaseTool]:
        """Return a copy of the internal name-to-tool mapping.

        Returns:
            Dictionary mapping tool names to BaseTool instances.
        """
        return dict(self._tools)
