from abc import ABC, abstractmethod


class BaseTool(ABC):
    """Abstract base class for tool implementations."""

    name: str
    description: str
    parameters: dict = {"type": "object", "properties": {}}

    @abstractmethod
    async def run(self, input: str, session_id: str = None, **kwargs) -> str:
        """Execute the tool with given input.

        Args:
            input: Tool input string.
            session_id: Optional session identifier for context.
            **kwargs: Additional keyword arguments.

        Returns:
            Tool execution result as string.
        """
        ...