"""Tests for MemoryExtractor - TDD implementation."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from agent_framework.interfaces.session import Message
from agent_framework.memory.extractor import MemoryExtractor, MemoryFact


class TestMemoryFact:
    """Test MemoryFact data class."""

    def test_memory_fact_creation(self):
        """Test creating a MemoryFact instance."""
        fact = MemoryFact(
            content="User prefers dark mode",
            metadata={"source": "conversation"},
            user_id="user123"
        )
        assert fact.content == "User prefers dark mode"
        assert fact.metadata == {"source": "conversation"}
        assert fact.user_id == "user123"

    def test_memory_fact_optional_user_id(self):
        """Test MemoryFact with optional user_id."""
        fact = MemoryFact(
            content="System maintenance scheduled",
            metadata={"type": "system"}
        )
        assert fact.user_id is None

    def test_memory_fact_default_metadata(self):
        """Test MemoryFact with default empty metadata."""
        fact = MemoryFact(content="Test fact")
        assert fact.metadata == {}
        assert fact.user_id is None


class TestMemoryExtractor:
    """Test MemoryExtractor class."""

    @pytest.fixture
    def mock_llm_gateway(self):
        """Create a mock LLM gateway."""
        mock = AsyncMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.fixture
    def extractor(self, mock_llm_gateway):
        """Create a MemoryExtractor instance with mock LLM."""
        return MemoryExtractor(llm_gateway=mock_llm_gateway, model_name="test-model")

    def test_extractor_initialization(self, mock_llm_gateway):
        """Test MemoryExtractor initialization."""
        extractor = MemoryExtractor(llm_gateway=mock_llm_gateway, model_name="test-model")
        assert extractor.llm == mock_llm_gateway
        assert extractor.model == "test-model"

    def test_extractor_default_model(self, mock_llm_gateway):
        """Test MemoryExtractor with default model name."""
        extractor = MemoryExtractor(llm_gateway=mock_llm_gateway)
        assert extractor.model == "light-model"

    @pytest.mark.asyncio
    async def test_extract_messages_to_facts(self, extractor, mock_llm_gateway):
        """Test extracting facts from messages."""
        # Prepare test messages
        messages = [
            Message(role="user", content="I prefer Python over Java", sender_id="user1"),
            Message(role="assistant", content="I'll remember your preference for Python", sender_id="assistant"),
            Message(role="user", content="My favorite framework is FastAPI", sender_id="user1")
        ]

        # Mock LLM response
        mock_llm_gateway.generate.return_value = '''
        [
            {
                "content": "User prefers Python over Java",
                "metadata": {"preference": "programming_language"},
                "user_id": "user1"
            },
            {
                "content": "User's favorite framework is FastAPI",
                "metadata": {"preference": "framework"},
                "user_id": "user1"
            }
        ]
        '''

        # Call extract method
        facts = await extractor.extract(messages)

        # Verify results
        assert len(facts) == 2
        assert isinstance(facts[0], MemoryFact)
        assert facts[0].content == "User prefers Python over Java"
        assert facts[0].metadata == {"preference": "programming_language"}
        assert facts[0].user_id == "user1"
        assert facts[1].content == "User's favorite framework is FastAPI"

        # Verify LLM was called
        mock_llm_gateway.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_empty_messages(self, extractor, mock_llm_gateway):
        """Test extracting from empty message list."""
        facts = await extractor.extract([])
        assert facts == []
        mock_llm_gateway.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_extract_no_facts_found(self, extractor, mock_llm_gateway):
        """Test when no facts can be extracted."""
        messages = [
            Message(role="user", content="Hello", sender_id="user1"),
            Message(role="assistant", content="Hi there!", sender_id="assistant")
        ]

        # Mock LLM response with empty array
        mock_llm_gateway.generate.return_value = "[]"

        facts = await extractor.extract(messages)
        assert facts == []

    @pytest.mark.asyncio
    async def test_extract_invalid_json_response(self, extractor, mock_llm_gateway):
        """Test handling invalid JSON response from LLM."""
        messages = [
            Message(role="user", content="Test message", sender_id="user1")
        ]

        # Mock invalid JSON response
        mock_llm_gateway.generate.return_value = "Invalid JSON response"

        with pytest.raises(ValueError) as exc_info:
            await extractor.extract(messages)
        assert "Invalid JSON response" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_is_important_message_true(self, extractor, mock_llm_gateway):
        """Test identifying important message."""
        message = Message(
            role="user",
            content="I have a severe allergy to peanuts",
            sender_id="user1"
        )

        # Mock LLM response indicating important
        mock_llm_gateway.generate.return_value = "true"

        result = await extractor.is_important(message)
        assert result is True

    @pytest.mark.asyncio
    async def test_is_important_message_false(self, extractor, mock_llm_gateway):
        """Test identifying unimportant message."""
        message = Message(
            role="user",
            content="The weather is nice today",
            sender_id="user1"
        )

        # Mock LLM response indicating not important
        mock_llm_gateway.generate.return_value = "false"

        result = await extractor.is_important(message)
        assert result is False

    @pytest.mark.asyncio
    async def test_is_important_invalid_response(self, extractor, mock_llm_gateway):
        """Test handling invalid response for importance check."""
        message = Message(
            role="user",
            content="Test message",
            sender_id="user1"
        )

        # Mock invalid response
        mock_llm_gateway.generate.return_value="maybe"

        with pytest.raises(ValueError) as exc_info:
            await extractor.is_important(message)
        assert "Invalid boolean response" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_extract_with_multiple_users(self, extractor, mock_llm_gateway):
        """Test extracting facts from multiple users."""
        messages = [
            Message(role="user", content="I'm Alice and I like cats", sender_id="alice"),
            Message(role="user", content="I'm Bob and I prefer dogs", sender_id="bob"),
            Message(role="assistant", content="I'll remember your preferences", sender_id="assistant")
        ]

        # Mock LLM response
        mock_llm_gateway.generate.return_value = '''
        [
            {
                "content": "Alice likes cats",
                "metadata": {"preference": "pet"},
                "user_id": "alice"
            },
            {
                "content": "Bob prefers dogs",
                "metadata": {"preference": "pet"},
                "user_id": "bob"
            }
        ]
        '''

        facts = await extractor.extract(messages)
        assert len(facts) == 2
        assert facts[0].user_id == "alice"
        assert facts[1].user_id == "bob"

    @pytest.mark.asyncio
    async def test_extract_with_metadata_extraction(self, extractor, mock_llm_gateway):
        """Test that metadata is properly extracted from LLM response."""
        messages = [
            Message(role="user", content="I work at Google as a software engineer", sender_id="user1")
        ]

        # Mock LLM response with complex metadata
        mock_llm_gateway.generate.return_value = '''
        [
            {
                "content": "User works at Google as software engineer",
                "metadata": {
                    "company": "Google",
                    "role": "software engineer",
                    "type": "employment"
                },
                "user_id": "user1"
            }
        ]
        '''

        facts = await extractor.extract(messages)
        assert len(facts) == 1
        assert facts[0].metadata["company"] == "Google"
        assert facts[0].metadata["role"] == "software engineer"
        assert facts[0].metadata["type"] == "employment"

    @pytest.mark.asyncio
    async def test_extract_preserves_message_order(self, extractor, mock_llm_gateway):
        """Test that extraction preserves order of facts from messages."""
        messages = [
            Message(role="user", content="First fact about me", sender_id="user1"),
            Message(role="user", content="Second fact about me", sender_id="user1")
        ]

        # Mock LLM response maintaining order
        mock_llm_gateway.generate.return_value = '''
        [
            {"content": "First fact", "metadata": {}, "user_id": "user1"},
            {"content": "Second fact", "metadata": {}, "user_id": "user1"}
        ]
        '''

        facts = await extractor.extract(messages)
        assert facts[0].content == "First fact"
        assert facts[1].content == "Second fact"

    @pytest.mark.asyncio
    async def test_extract_with_system_messages(self, extractor, mock_llm_gateway):
        """Test that system messages are included in extraction context."""
        messages = [
            Message(role="system", content="You are a helpful assistant", sender_id="system"),
            Message(role="user", content="I need help with Python", sender_id="user1"),
            Message(role="assistant", content="I can help with Python", sender_id="assistant")
        ]

        # Mock LLM response
        mock_llm_gateway.generate.return_value = '''
        [
            {"content": "User needs help with Python", "metadata": {"topic": "programming"}, "user_id": "user1"}
        ]
        '''

        facts = await extractor.extract(messages)
        assert len(facts) == 1
        assert "Python" in facts[0].content

    @pytest.mark.asyncio
    async def test_extract_with_empty_content(self, extractor, mock_llm_gateway):
        """Test handling messages with empty content."""
        messages = [
            Message(role="user", content="", sender_id="user1"),
            Message(role="assistant", content="I didn't receive your message", sender_id="assistant")
        ]

        # Mock LLM response - should handle gracefully
        mock_llm_gateway.generate.return_value = "[]"

        facts = await extractor.extract(messages)
        assert facts == []

    @pytest.mark.asyncio
    async def test_extract_with_very_long_message(self, extractor, mock_llm_gateway):
        """Test extraction with very long message content."""
        long_content = "A" * 10000  # 10k characters
        messages = [
            Message(role="user", content=long_content, sender_id="user1")
        ]

        # Mock LLM response
        mock_llm_gateway.generate.return_value = '''
        [
            {"content": "User provided long text", "metadata": {"length": "long"}, "user_id": "user1"}
        ]
        '''

        facts = await extractor.extract(messages)
        assert len(facts) == 1

    @pytest.mark.asyncio
    async def test_extract_with_special_characters(self, extractor, mock_llm_gateway):
        """Test extraction with special characters in content."""
        messages = [
            Message(role="user", content="I love Python & JavaScript! @#$%^&*()", sender_id="user1")
        ]

        # Mock LLM response
        mock_llm_gateway.generate.return_value = '''
        [
            {"content": "User loves Python and JavaScript", "metadata": {"languages": ["Python", "JavaScript"]}, "user_id": "user1"}
        ]
        '''

        facts = await extractor.extract(messages)
        assert len(facts) == 1
        assert "Python" in facts[0].content

    @pytest.mark.asyncio
    async def test_extract_with_conversation_context(self, extractor, mock_llm_gateway):
        """Test that extraction considers full conversation context."""
        messages = [
            Message(role="user", content="What's the capital of France?", sender_id="user1"),
            Message(role="assistant", content="The capital of France is Paris.", sender_id="assistant"),
            Message(role="user", content="I'm planning to visit there next month", sender_id="user1")
        ]

        # Mock LLM response - should capture travel plans
        mock_llm_gateway.generate.return_value = '''
        [
            {
                "content": "User plans to visit Paris next month",
                "metadata": {"destination": "Paris", "timing": "next month", "type": "travel_plan"},
                "user_id": "user1"
            }
        ]
        '''

        facts = await extractor.extract(messages)
        assert len(facts) == 1
        assert "Paris" in facts[0].content
        assert facts[0].metadata["type"] == "travel_plan"

    @pytest.mark.asyncio
    async def test_extract_with_multiple_facts_per_message(self, extractor, mock_llm_gateway):
        """Test extracting multiple facts from a single message."""
        messages = [
            Message(role="user", content="I'm a software engineer at Google, I prefer Python, and I live in San Francisco", sender_id="user1")
        ]

        # Mock LLM response with multiple facts
        mock_llm_gateway.generate.return_value = '''
        [
            {"content": "User is a software engineer at Google", "metadata": {"company": "Google"}, "user_id": "user1"},
            {"content": "User prefers Python", "metadata": {"preference": "language"}, "user_id": "user1"},
            {"content": "User lives in San Francisco", "metadata": {"location": "San Francisco"}, "user_id": "user1"}
        ]
        '''

        facts = await extractor.extract(messages)
        assert len(facts) == 3
        assert any("Google" in f.content for f in facts)
        assert any("Python" in f.content for f in facts)
        assert any("San Francisco" in f.content for f in facts)

    @pytest.mark.asyncio
    async def test_is_important_with_context(self, extractor, mock_llm_gateway):
        """Test importance detection with message context."""
        # Previous context: user mentioned allergies
        message = Message(
            role="user",
            content="I'm feeling dizzy after eating at that restaurant",
            sender_id="user1"
        )

        # Mock LLM response - should be important given context
        mock_llm_gateway.generate.return_value = "true"

        result = await extractor.is_important(message)
        assert result is True

    @pytest.mark.asyncio
    async def test_extract_with_different_roles(self, extractor, mock_llm_gateway):
        """Test extraction handles different message roles appropriately."""
        messages = [
            Message(role="system", content="Be helpful and remember user preferences", sender_id="system"),
            Message(role="user", content="I prefer dark mode interfaces", sender_id="user1"),
            Message(role="assistant", content="I'll remember that preference", sender_id="assistant"),
            Message(role="user", content="Also, I'm color blind", sender_id="user1")
        ]

        # Mock LLM response
        mock_llm_gateway.generate.return_value = '''
        [
            {"content": "User prefers dark mode interfaces", "metadata": {"preference": "UI"}, "user_id": "user1"},
            {"content": "User is color blind", "metadata": {"accessibility": "color_blindness"}, "user_id": "user1"}
        ]
        '''

        facts = await extractor.extract(messages)
        assert len(facts) == 2

    @pytest.mark.asyncio
    async def test_extract_with_empty_metadata(self, extractor, mock_llm_gateway):
        """Test extraction when LLM returns empty metadata."""
        messages = [
            Message(role="user", content="Simple fact", sender_id="user1")
        ]

        # Mock LLM response with empty metadata
        mock_llm_gateway.generate.return_value = '''
        [
            {"content": "Simple fact", "metadata": {}, "user_id": "user1"}
        ]
        '''

        facts = await extractor.extract(messages)
        assert len(facts) == 1
        assert facts[0].metadata == {}

    @pytest.mark.asyncio
    async def test_extract_with_null_user_id(self, extractor, mock_llm_gateway):
        """Test extraction when LLM returns null user_id."""
        messages = [
            Message(role="user", content="General knowledge", sender_id="user1")
        ]

        # Mock LLM response with null user_id
        mock_llm_gateway.generate.return_value = '''
        [
            {"content": "General knowledge", "metadata": {}, "user_id": null}
        ]
        '''

        facts = await extractor.extract(messages)
        assert len(facts) == 1
        assert facts[0].user_id is None

    @pytest.mark.asyncio
    async def test_extract_with_missing_fields(self, extractor, mock_llm_gateway):
        """Test extraction handles missing optional fields."""
        messages = [
            Message(role="user", content="Test message", sender_id="user1")
        ]

        # Mock LLM response with missing metadata and user_id
        mock_llm_gateway.generate.return_value = '''
        [
            {"content": "Test fact"}
        ]
        '''

        facts = await extractor.extract(messages)
        assert len(facts) == 1
        assert facts[0].content == "Test fact"
        assert facts[0].metadata == {}
        assert facts[0].user_id is None

    @pytest.mark.asyncio
    async def test_extract_with_invalid_fact_structure(self, extractor, mock_llm_gateway):
        """Test extraction handles invalid fact structure."""
        messages = [
            Message(role="user", content="Test message", sender_id="user1")
        ]

        # Mock LLM response with invalid structure
        mock_llm_gateway.generate.return_value = '''
        [
            {"invalid_field": "value"}
        ]
        '''

        with pytest.raises(ValueError) as exc_info:
            await extractor.extract(messages)
        assert "Missing required field" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_extract_with_non_list_response(self, extractor, mock_llm_gateway):
        """Test extraction handles non-list response from LLM."""
        messages = [
            Message(role="user", content="Test message", sender_id="user1")
        ]

        # Mock LLM response that's not a list
        mock_llm_gateway.generate.return_value = '''
        {"content": "Not a list"}
        '''

        with pytest.raises(ValueError) as exc_info:
            await extractor.extract(messages)
        assert "Invalid JSON response" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_extract_with_malformed_json(self, extractor, mock_llm_gateway):
        """Test extraction handles malformed JSON."""
        messages = [
            Message(role="user", content="Test message", sender_id="user1")
        ]

        # Mock malformed JSON
        mock_llm_gateway.generate.return_value = '''
        [
            {"content": "Test fact", "metadata": {}, "user_id": "user1"}
        '''  # Missing closing bracket

        with pytest.raises(ValueError) as exc_info:
            await extractor.extract(messages)
        assert "Invalid JSON" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_is_important_with_different_responses(self, extractor, mock_llm_gateway):
        """Test importance detection with various LLM responses."""
        message = Message(role="user", content="Test", sender_id="user1")

        # Test various response formats
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("yes", True),  # Should handle common alternatives
            ("no", False),
        ]

        for response, expected in test_cases:
            mock_llm_gateway.generate.return_value = response
            result = await extractor.is_important(message)
            assert result == expected, f"Failed for response: {response}"

    @pytest.mark.asyncio
    async def test_extract_with_conversation_history(self, extractor, mock_llm_gateway):
        """Test extraction considers full conversation history."""
        # Long conversation history
        messages = []
        for i in range(20):
            messages.append(Message(role="user", content=f"Message {i}", sender_id="user1"))
            messages.append(Message(role="assistant", content=f"Response {i}", sender_id="assistant"))

        # Mock LLM response
        mock_llm_gateway.generate.return_value = '''
        [
            {"content": "User had long conversation", "metadata": {"message_count": 40}, "user_id": "user1"}
        ]
        '''

        facts = await extractor.extract(messages)
        assert len(facts) == 1

    @pytest.mark.asyncio
    async def test_extract_with_different_users_multiple_sessions(self, extractor, mock_llm_gateway):
        """Test extraction with users from different sessions."""
        messages = [
            Message(role="user", content="I'm user1 from session A", sender_id="user1"),
            Message(role="user", content="I'm user2 from session B", sender_id="user2"),
            Message(role="assistant", content="Acknowledged", sender_id="assistant")
        ]

        # Mock LLM response
        mock_llm_gateway.generate.return_value = '''
        [
            {"content": "User1 from session A", "metadata": {"session": "A"}, "user_id": "user1"},
            {"content": "User2 from session B", "metadata": {"session": "B"}, "user_id": "user2"}
        ]
        '''

        facts = await extractor.extract(messages)
        assert len(facts) == 2
        user_ids = {f.user_id for f in facts}
        assert "user1" in user_ids
        assert "user2" in user_ids