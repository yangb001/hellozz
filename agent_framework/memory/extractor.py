"""MemoryExtractor - Extracts facts and memories from conversation messages.

This module uses LLM to analyze conversation messages and extract
important facts, user preferences, and other long-term memory-worthy
information.

参考：详细设计.md 第6.4节
"""
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from agent_framework.interfaces.session import Message
from agent_framework.infrastructure.llm_gateway import LLMGateway

logger = logging.getLogger(__name__)


@dataclass
class MemoryFact:
    """Represents an extracted fact from conversation.

    Attributes:
        content: The textual content of the extracted fact.
        metadata: Additional metadata about the fact (e.g., type, source).
        user_id: Optional user ID if the fact is user-specific.
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None


class MemoryExtractor:
    """Extracts facts and memories from conversation messages using LLM.

    Uses a lightweight LLM to analyze conversation messages and extract
    important facts, user preferences, and other information worth storing
    in long-term memory. Also provides importance scoring for individual
    messages to support smart extraction triggers.

    Attributes:
        llm: The LLM gateway instance for generating completions.
        model: The model alias to use for extraction.
    """

    # Prompt for extracting facts from conversation messages
    EXTRACT_PROMPT = """Analyze the following conversation messages and extract important facts about users, their preferences, personal information, and any other details worth remembering long-term.

Conversation messages:
{messages}

Extract facts as a JSON array. Each fact should have:
- "content": A clear, concise statement of the fact
- "metadata": An object with relevant metadata (e.g., type, category, context)
- "user_id": The user ID who the fact is about (null if general/system knowledge)

Rules:
- Only extract facts that are worth remembering long-term
- Combine related information into single coherent facts
- Extract user preferences, personal details, important plans, and notable information
- Do NOT extract trivial exchanges, greetings, or system messages
- If no facts are found, return an empty array []
- Return ONLY valid JSON, no other text

Response (JSON array):"""

    # Prompt for checking if a message is important
    IMPORTANCE_PROMPT = """Determine if the following message contains information worth remembering long-term. Consider:
- Personal preferences or opinions
- Important facts about the user
- Plans, goals, or commitments
- Medical, dietary, or accessibility information
- Professional or personal details
- Any information that would be useful to remember in future conversations

Message:
{message}

Respond with ONLY "true" if the message contains important information worth remembering, or "false" if it's trivial/not worth remembering.

Response (true/false):"""

    def __init__(self, llm_gateway: Optional[LLMGateway] = None, model_name: str = "light-model"):
        """Initialize the MemoryExtractor.

        Args:
            llm_gateway: The LLM gateway instance for generating completions.
                        If None, extraction operations will be no-ops.
            model_name: The model alias to use for extraction (default: "light-model").
        """
        self.llm = llm_gateway
        self.model = model_name

    async def extract(self, messages: List[Message]) -> List[MemoryFact]:
        """Extract facts from a list of conversation messages.

        Analyzes the conversation messages using LLM to identify and extract
        important facts about users, their preferences, and other information
        worth storing in long-term memory.

        Args:
            messages: List of Message objects from the conversation.

        Returns:
            List of MemoryFact objects extracted from the messages.

        Raises:
            ValueError: If the LLM response is not valid JSON or has invalid structure.
        """
        if not messages or self.llm is None:
            return []

        # Format messages for the prompt
        formatted_messages = self._format_messages(messages)
        prompt = self.EXTRACT_PROMPT.format(messages=formatted_messages)

        # Call LLM to extract facts
        response = await self.llm.generate(prompt, model=self.model)

        # Parse and validate response
        facts = self._parse_extract_response(response)

        logger.debug(f"Extracted {len(facts)} facts from {len(messages)} messages")
        return facts

    async def is_important(self, msg: Message) -> bool:
        """Check if a message contains important information worth remembering.

        Uses LLM to analyze a single message and determine if it contains
        information that should trigger immediate long-term memory extraction.

        Args:
            msg: The Message object to check for importance.

        Returns:
            True if the message is important, False otherwise.

        Raises:
            ValueError: If the LLM response is not a valid boolean string.
        """
        if self.llm is None:
            return False

        # Format the message for the prompt
        formatted_msg = f"[{msg.role}] {msg.content}"
        prompt = self.IMPORTANCE_PROMPT.format(message=formatted_msg)

        # Call LLM to check importance
        response = await self.llm.generate(prompt, model=self.model)

        # Parse boolean response
        result = self._parse_boolean_response(response)

        logger.debug(f"Message importance check: {result} (content: {msg.content[:50]}...)")
        return result

    def _format_messages(self, messages: List[Message]) -> str:
        """Format messages into a readable string for the prompt.

        Args:
            messages: List of Message objects to format.

        Returns:
            Formatted string representation of the messages.
        """
        formatted = []
        for msg in messages:
            sender = msg.sender_id or msg.role
            formatted.append(f"[{msg.role}] ({sender}): {msg.content}")
        return "\n".join(formatted)

    def _parse_extract_response(self, response: str) -> List[MemoryFact]:
        """Parse and validate the LLM extraction response.

        Args:
            response: Raw response string from the LLM.

        Returns:
            List of MemoryFact objects parsed from the response.

        Raises:
            ValueError: If the response is not valid JSON or has invalid structure.
        """
        # Clean response - strip whitespace and find JSON array
        response = response.strip()

        # Try to find JSON array in the response
        start_idx = response.find('[')
        end_idx = response.rfind(']')

        if start_idx == -1 or end_idx == -1:
            raise ValueError(f"Invalid JSON response: no JSON array found in: {response[:200]}")

        json_str = response[start_idx:end_idx + 1]

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}. Response: {response[:200]}")

        if not isinstance(data, list):
            raise ValueError(f"Expected list response, got {type(data).__name__}")

        # Parse each fact
        facts = []
        for item in data:
            if not isinstance(item, dict):
                raise ValueError(f"Expected dict in list, got {type(item).__name__}")

            # Validate required field
            if "content" not in item:
                raise ValueError(f"Missing required field 'content' in fact: {item}")

            fact = MemoryFact(
                content=item["content"],
                metadata=item.get("metadata", {}),
                user_id=item.get("user_id")
            )
            facts.append(fact)

        return facts

    def _parse_boolean_response(self, response: str) -> bool:
        """Parse a boolean response from the LLM.

        Args:
            response: Raw response string from the LLM.

        Returns:
            Boolean value parsed from the response.

        Raises:
            ValueError: If the response cannot be parsed as a boolean.
        """
        cleaned = response.strip().lower()

        # Accept common boolean representations
        true_values = {"true", "yes", "1"}
        false_values = {"false", "no", "0"}

        if cleaned in true_values:
            return True
        elif cleaned in false_values:
            return False
        else:
            raise ValueError(f"Invalid boolean response: '{response}'. Expected 'true' or 'false'.")
