"""Independent test cases for MemoryExtractor implementation.

This module contains independent verification tests for the MemoryExtractor
and MemoryFact classes, following the detailed design specification in section 6.4.

Test categories:
1. MemoryFact data class integrity
2. MemoryExtractor initialization
3. extract method behavior
4. is_important method behavior
5. Response parsing
6. Boundary conditions and error handling
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import List, Dict, Any

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_framework.interfaces.session import Message
from agent_framework.memory.extractor import MemoryExtractor, MemoryFact


class TestMemoryFactDataClass:
    """Independent tests for MemoryFact data class."""

    def test_memory_fact_has_content_field(self):
        """MemoryFact must have content field."""
        fact = MemoryFact(content="test fact")
        assert fact.content == "test fact"

    def test_memory_fact_has_metadata_field(self):
        """MemoryFact must have metadata field."""
        fact = MemoryFact(content="test", metadata={"key": "value"})
        assert fact.metadata == {"key": "value"}

    def test_memory_fact_has_user_id_field(self):
        """MemoryFact must have user_id field."""
        fact = MemoryFact(content="test", user_id="user123")
        assert fact.user_id == "user123"

    def test_memory_fact_default_metadata_empty_dict(self):
        """MemoryFact metadata should default to empty dict."""
        fact = MemoryFact(content="test")
        assert fact.metadata == {}

    def test_memory_fact_default_user_id_none(self):
        """MemoryFact user_id should default to None."""
        fact = MemoryFact(content="test")
        assert fact.user_id is None

    def test_memory_fact_is_dataclass(self):
        """MemoryFact should be a dataclass."""
        import dataclasses
        assert dataclasses.is_dataclass(MemoryFact)

    def test_memory_fact_content_must_be_string(self):
        """MemoryFact content should accept string."""
        fact = MemoryFact(content="test content")
        assert isinstance(fact.content, str)

    def test_memory_fact_metadata_must_be_dict(self):
        """MemoryFact metadata should accept dict."""
        metadata = {"type": "preference", "source": "conversation"}
        fact = MemoryFact(content="test", metadata=metadata)
        assert isinstance(fact.metadata, dict)

    def test_memory_fact_user_id_must_be_string_or_none(self):
        """MemoryFact user_id should accept string or None."""
        fact1 = MemoryFact(content="test", user_id="user1")
        fact2 = MemoryFact(content="test", user_id=None)
        assert isinstance(fact1.user_id, str)
        assert fact2.user_id is None

    def test_memory_fact_with_complex_metadata(self):
        """MemoryFact should accept complex nested metadata."""
        metadata = {
            "type": "employment",
            "details": {
                "company": "Google",
                "role": "engineer"
            },
            "tags": ["work", "tech"]
        }
        fact = MemoryFact(content="test", metadata=metadata)
        assert fact.metadata["details"]["company"] == "Google"
        assert "work" in fact.metadata["tags"]

    def test_memory_fact_with_empty_content(self):
        """MemoryFact should accept empty string content."""
        fact = MemoryFact(content="")
        assert fact.content == ""


class TestMemoryExtractorInitialization:
    """Independent tests for MemoryExtractor initialization."""

    def test_extractor_stores_llm_gateway(self):
        """MemoryExtractor should store llm_gateway reference."""
        mock_llm = MagicMock()
        extractor = MemoryExtractor(llm_gateway=mock_llm)
        assert extractor.llm is mock_llm

    def test_extractor_stores_model_name(self):
        """MemoryExtractor should store model name."""
        mock_llm = MagicMock()
        extractor = MemoryExtractor(llm_gateway=mock_llm, model_name="test-model")
        assert extractor.model == "test-model"

    def test_extractor_default_model_name(self):
        """MemoryExtractor should default to 'light-model'."""
        mock_llm = MagicMock()
        extractor = MemoryExtractor(llm_gateway=mock_llm)
        assert extractor.model == "light-model"

    def test_extractor_has_extract_method(self):
        """MemoryExtractor must have extract method."""
        assert hasattr(MemoryExtractor, 'extract')
        assert callable(getattr(MemoryExtractor, 'extract'))

    def test_extractor_has_is_important_method(self):
        """MemoryExtractor must have is_important method."""
        assert hasattr(MemoryExtractor, 'is_important')
        assert callable(getattr(MemoryExtractor, 'is_important'))

    def test_extractor_has_extract_prompt(self):
        """MemoryExtractor should have EXTRACT_PROMPT class attribute."""
        assert hasattr(MemoryExtractor, 'EXTRACT_PROMPT')
        assert isinstance(MemoryExtractor.EXTRACT_PROMPT, str)

    def test_extractor_has_importance_prompt(self):
        """MemoryExtractor should have IMPORTANCE_PROMPT class attribute."""
        assert hasattr(MemoryExtractor, 'IMPORTANCE_PROMPT')
        assert isinstance(MemoryExtractor.IMPORTANCE_PROMPT, str)


class TestMemoryExtractorExtractMethod:
    """Independent tests for extract method."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM gateway."""
        mock = AsyncMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.fixture
    def extractor(self, mock_llm):
        """Create MemoryExtractor with mock LLM."""
        return MemoryExtractor(llm_gateway=mock_llm, model_name="test-model")

    @pytest.mark.asyncio
    async def test_extract_returns_list(self, extractor, mock_llm):
        """extract should return a list."""
        messages = [Message(role="user", content="I like Python", sender_id="user1")]
        mock_llm.generate.return_value = '[{"content": "User likes Python"}]'

        result = await extractor.extract(messages)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_extract_returns_memory_facts(self, extractor, mock_llm):
        """extract should return list of MemoryFact."""
        messages = [Message(role="user", content="I like Python", sender_id="user1")]
        mock_llm.generate.return_value = '[{"content": "User likes Python"}]'

        result = await extractor.extract(messages)
        assert all(isinstance(f, MemoryFact) for f in result)

    @pytest.mark.asyncio
    async def test_extract_empty_messages_returns_empty(self, extractor, mock_llm):
        """extract with empty messages should return empty list."""
        result = await extractor.extract([])
        assert result == []
        mock_llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_extract_calls_llm_once(self, extractor, mock_llm):
        """extract should call LLM exactly once."""
        messages = [Message(role="user", content="test", sender_id="user1")]
        mock_llm.generate.return_value = '[]'

        await extractor.extract(messages)
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_uses_correct_model(self, extractor, mock_llm):
        """extract should use the configured model."""
        messages = [Message(role="user", content="test", sender_id="user1")]
        mock_llm.generate.return_value = '[]'

        await extractor.extract(messages)
        call_args = mock_llm.generate.call_args
        assert call_args[1].get('model') == 'test-model' or call_args[0][1] == 'test-model'

    @pytest.mark.asyncio
    async def test_extract_parses_single_fact(self, extractor, mock_llm):
        """extract should parse single fact correctly."""
        messages = [Message(role="user", content="I live in Paris", sender_id="user1")]
        mock_llm.generate.return_value = '''
        [
            {
                "content": "User lives in Paris",
                "metadata": {"location": "Paris"},
                "user_id": "user1"
            }
        ]
        '''

        facts = await extractor.extract(messages)
        assert len(facts) == 1
        assert facts[0].content == "User lives in Paris"
        assert facts[0].metadata["location"] == "Paris"
        assert facts[0].user_id == "user1"

    @pytest.mark.asyncio
    async def test_extract_parses_multiple_facts(self, extractor, mock_llm):
        """extract should parse multiple facts correctly."""
        messages = [Message(role="user", content="I like cats and dogs", sender_id="user1")]
        mock_llm.generate.return_value = '''
        [
            {"content": "User likes cats", "metadata": {}, "user_id": "user1"},
            {"content": "User likes dogs", "metadata": {}, "user_id": "user1"}
        ]
        '''

        facts = await extractor.extract(messages)
        assert len(facts) == 2

    @pytest.mark.asyncio
    async def test_extract_handles_empty_array_response(self, extractor, mock_llm):
        """extract should handle empty array response."""
        messages = [Message(role="user", content="Hello", sender_id="user1")]
        mock_llm.generate.return_value = '[]'

        facts = await extractor.extract(messages)
        assert facts == []

    @pytest.mark.asyncio
    async def test_extract_invalid_json_raises_error(self, extractor, mock_llm):
        """extract should raise ValueError for invalid JSON."""
        messages = [Message(role="user", content="test", sender_id="user1")]
        mock_llm.generate.return_value = 'not valid json'

        with pytest.raises(ValueError):
            await extractor.extract(messages)

    @pytest.mark.asyncio
    async def test_extract_missing_content_raises_error(self, extractor, mock_llm):
        """extract should raise ValueError when content field is missing."""
        messages = [Message(role="user", content="test", sender_id="user1")]
        mock_llm.generate.return_value = '[{"metadata": {}, "user_id": "user1"}]'

        with pytest.raises(ValueError) as exc_info:
            await extractor.extract(messages)
        assert "content" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_extract_non_list_response_raises_error(self, extractor, mock_llm):
        """extract should raise ValueError when response is not a list."""
        messages = [Message(role="user", content="test", sender_id="user1")]
        mock_llm.generate.return_value = '{"content": "not a list"}'

        with pytest.raises(ValueError):
            await extractor.extract(messages)

    @pytest.mark.asyncio
    async def test_extract_with_multiple_messages(self, extractor, mock_llm):
        """extract should handle multiple messages."""
        messages = [
            Message(role="user", content="I like Python", sender_id="user1"),
            Message(role="assistant", content="I'll remember that", sender_id="assistant"),
            Message(role="user", content="I also like FastAPI", sender_id="user1")
        ]
        mock_llm.generate.return_value = '''
        [
            {"content": "User likes Python", "metadata": {}, "user_id": "user1"},
            {"content": "User likes FastAPI", "metadata": {}, "user_id": "user1"}
        ]
        '''

        facts = await extractor.extract(messages)
        assert len(facts) == 2


class TestMemoryExtractorIsImportantMethod:
    """Independent tests for is_important method."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM gateway."""
        mock = AsyncMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.fixture
    def extractor(self, mock_llm):
        """Create MemoryExtractor with mock LLM."""
        return MemoryExtractor(llm_gateway=mock_llm, model_name="test-model")

    @pytest.mark.asyncio
    async def test_is_important_returns_bool(self, extractor, mock_llm):
        """is_important should return a boolean."""
        msg = Message(role="user", content="test", sender_id="user1")
        mock_llm.generate.return_value = "true"

        result = await extractor.is_important(msg)
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_is_important_true_response(self, extractor, mock_llm):
        """is_important should return True for 'true' response."""
        msg = Message(role="user", content="I have a peanut allergy", sender_id="user1")
        mock_llm.generate.return_value = "true"

        result = await extractor.is_important(msg)
        assert result is True

    @pytest.mark.asyncio
    async def test_is_important_false_response(self, extractor, mock_llm):
        """is_important should return False for 'false' response."""
        msg = Message(role="user", content="Hello", sender_id="user1")
        mock_llm.generate.return_value = "false"

        result = await extractor.is_important(msg)
        assert result is False

    @pytest.mark.asyncio
    async def test_is_important_case_insensitive(self, extractor, mock_llm):
        """is_important should handle case-insensitive responses."""
        msg = Message(role="user", content="test", sender_id="user1")

        for response in ["True", "TRUE", "true"]:
            mock_llm.generate.return_value = response
            result = await extractor.is_important(msg)
            assert result is True

        for response in ["False", "FALSE", "false"]:
            mock_llm.generate.return_value = response
            result = await extractor.is_important(msg)
            assert result is False

    @pytest.mark.asyncio
    async def test_is_important_yes_response(self, extractor, mock_llm):
        """is_important should accept 'yes' as True."""
        msg = Message(role="user", content="test", sender_id="user1")
        mock_llm.generate.return_value = "yes"

        result = await extractor.is_important(msg)
        assert result is True

    @pytest.mark.asyncio
    async def test_is_important_no_response(self, extractor, mock_llm):
        """is_important should accept 'no' as False."""
        msg = Message(role="user", content="test", sender_id="user1")
        mock_llm.generate.return_value = "no"

        result = await extractor.is_important(msg)
        assert result is False

    @pytest.mark.asyncio
    async def test_is_important_calls_llm_once(self, extractor, mock_llm):
        """is_important should call LLM exactly once."""
        msg = Message(role="user", content="test", sender_id="user1")
        mock_llm.generate.return_value = "true"

        await extractor.is_important(msg)
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_is_important_invalid_response_raises_error(self, extractor, mock_llm):
        """is_important should raise ValueError for invalid response."""
        msg = Message(role="user", content="test", sender_id="user1")
        mock_llm.generate.return_value = "maybe"

        with pytest.raises(ValueError):
            await extractor.is_important(msg)

    @pytest.mark.asyncio
    async def test_is_important_with_whitespace_response(self, extractor, mock_llm):
        """is_important should handle whitespace in response."""
        msg = Message(role="user", content="test", sender_id="user1")
        mock_llm.generate.return_value = "  true  "

        result = await extractor.is_important(msg)
        assert result is True


class TestMemoryExtractorFormatMessages:
    """Independent tests for _format_messages helper."""

    @pytest.fixture
    def extractor(self):
        """Create MemoryExtractor with mock LLM."""
        return MemoryExtractor(llm_gateway=MagicMock())

    def test_format_messages_with_role(self, extractor):
        """_format_messages should include role."""
        messages = [Message(role="user", content="Hello", sender_id="user1")]
        result = extractor._format_messages(messages)
        assert "user" in result

    def test_format_messages_with_content(self, extractor):
        """_format_messages should include content."""
        messages = [Message(role="user", content="Hello World", sender_id="user1")]
        result = extractor._format_messages(messages)
        assert "Hello World" in result

    def test_format_messages_with_sender_id(self, extractor):
        """_format_messages should include sender_id."""
        messages = [Message(role="user", content="Hello", sender_id="alice")]
        result = extractor._format_messages(messages)
        assert "alice" in result

    def test_format_messages_multiple_messages(self, extractor):
        """_format_messages should handle multiple messages."""
        messages = [
            Message(role="user", content="Hello", sender_id="user1"),
            Message(role="assistant", content="Hi", sender_id="assistant")
        ]
        result = extractor._format_messages(messages)
        assert "Hello" in result
        assert "Hi" in result

    def test_format_messages_empty_list(self, extractor):
        """_format_messages should handle empty list."""
        result = extractor._format_messages([])
        assert result == ""


class TestMemoryExtractorParseExtractResponse:
    """Independent tests for _parse_extract_response helper."""

    @pytest.fixture
    def extractor(self):
        """Create MemoryExtractor with mock LLM."""
        return MemoryExtractor(llm_gateway=MagicMock())

    def test_parse_valid_json_array(self, extractor):
        """_parse_extract_response should parse valid JSON array."""
        response = '[{"content": "test fact", "metadata": {}, "user_id": "user1"}]'
        facts = extractor._parse_extract_response(response)
        assert len(facts) == 1
        assert facts[0].content == "test fact"

    def test_parse_empty_array(self, extractor):
        """_parse_extract_response should handle empty array."""
        response = '[]'
        facts = extractor._parse_extract_response(response)
        assert facts == []

    def test_parse_with_extra_text(self, extractor):
        """_parse_extract_response should find JSON in extra text."""
        response = 'Here are the facts: [{"content": "test"}] end'
        facts = extractor._parse_extract_response(response)
        assert len(facts) == 1

    def test_parse_invalid_json_raises_error(self, extractor):
        """_parse_extract_response should raise ValueError for invalid JSON."""
        with pytest.raises(ValueError):
            extractor._parse_extract_response("not json")

    def test_parse_missing_content_raises_error(self, extractor):
        """_parse_extract_response should raise ValueError for missing content."""
        response = '[{"metadata": {}}]'
        with pytest.raises(ValueError) as exc_info:
            extractor._parse_extract_response(response)
        assert "content" in str(exc_info.value).lower()

    def test_parse_non_list_raises_error(self, extractor):
        """_parse_extract_response should raise ValueError for non-list."""
        response = '{"content": "not a list"}'
        with pytest.raises(ValueError):
            extractor._parse_extract_response(response)

    def test_parse_non_dict_in_list_raises_error(self, extractor):
        """_parse_extract_response should raise ValueError for non-dict items."""
        response = '["not a dict"]'
        with pytest.raises(ValueError):
            extractor._parse_extract_response(response)


class TestMemoryExtractorParseBooleanResponse:
    """Independent tests for _parse_boolean_response helper."""

    @pytest.fixture
    def extractor(self):
        """Create MemoryExtractor with mock LLM."""
        return MemoryExtractor(llm_gateway=MagicMock())

    def test_parse_true(self, extractor):
        """_parse_boolean_response should parse 'true'."""
        assert extractor._parse_boolean_response("true") is True

    def test_parse_false(self, extractor):
        """_parse_boolean_response should parse 'false'."""
        assert extractor._parse_boolean_response("false") is False

    def test_parse_case_insensitive(self, extractor):
        """_parse_boolean_response should be case-insensitive."""
        assert extractor._parse_boolean_response("True") is True
        assert extractor._parse_boolean_response("FALSE") is False

    def test_parse_yes(self, extractor):
        """_parse_boolean_response should accept 'yes'."""
        assert extractor._parse_boolean_response("yes") is True

    def test_parse_no(self, extractor):
        """_parse_boolean_response should accept 'no'."""
        assert extractor._parse_boolean_response("no") is False

    def test_parse_1(self, extractor):
        """_parse_boolean_response should accept '1'."""
        assert extractor._parse_boolean_response("1") is True

    def test_parse_0(self, extractor):
        """_parse_boolean_response should accept '0'."""
        assert extractor._parse_boolean_response("0") is False

    def test_parse_whitespace(self, extractor):
        """_parse_boolean_response should handle whitespace."""
        assert extractor._parse_boolean_response("  true  ") is True

    def test_parse_invalid_raises_error(self, extractor):
        """_parse_boolean_response should raise ValueError for invalid input."""
        with pytest.raises(ValueError):
            extractor._parse_boolean_response("maybe")

    def test_parse_empty_raises_error(self, extractor):
        """_parse_boolean_response should raise ValueError for empty string."""
        with pytest.raises(ValueError):
            extractor._parse_boolean_response("")


class TestMemoryExtractorBoundaryConditions:
    """Independent tests for boundary conditions."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM gateway."""
        mock = AsyncMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.fixture
    def extractor(self, mock_llm):
        """Create MemoryExtractor with mock LLM."""
        return MemoryExtractor(llm_gateway=mock_llm)

    @pytest.mark.asyncio
    async def test_extract_with_long_message(self, extractor, mock_llm):
        """extract should handle long messages."""
        long_content = "A" * 10000
        messages = [Message(role="user", content=long_content, sender_id="user1")]
        mock_llm.generate.return_value = '[{"content": "long message"}]'

        facts = await extractor.extract(messages)
        assert len(facts) == 1

    @pytest.mark.asyncio
    async def test_extract_with_special_characters(self, extractor, mock_llm):
        """extract should handle special characters."""
        messages = [Message(role="user", content="I love Python & JS! @#$%", sender_id="user1")]
        mock_llm.generate.return_value = '[{"content": "User loves Python and JS"}]'

        facts = await extractor.extract(messages)
        assert len(facts) == 1

    @pytest.mark.asyncio
    async def test_extract_with_unicode(self, extractor, mock_llm):
        """extract should handle unicode characters."""
        messages = [Message(role="user", content="我喜欢Python", sender_id="user1")]
        mock_llm.generate.return_value = '[{"content": "User likes Python"}]'

        facts = await extractor.extract(messages)
        assert len(facts) == 1

    @pytest.mark.asyncio
    async def test_extract_with_empty_content_message(self, extractor, mock_llm):
        """extract should handle messages with empty content."""
        messages = [
            Message(role="user", content="", sender_id="user1"),
            Message(role="assistant", content="I didn't get that", sender_id="assistant")
        ]
        mock_llm.generate.return_value = '[]'

        facts = await extractor.extract(messages)
        assert facts == []

    @pytest.mark.asyncio
    async def test_is_important_with_long_message(self, extractor, mock_llm):
        """is_important should handle long messages."""
        long_content = "A" * 10000
        msg = Message(role="user", content=long_content, sender_id="user1")
        mock_llm.generate.return_value = "false"

        result = await extractor.is_important(msg)
        assert result is False

    @pytest.mark.asyncio
    async def test_extract_with_json_in_content(self, extractor, mock_llm):
        """extract should handle messages containing JSON-like content."""
        messages = [Message(role="user", content='I like {"key": "value"}', sender_id="user1")]
        mock_llm.generate.return_value = '[{"content": "User likes JSON"}]'

        facts = await extractor.extract(messages)
        assert len(facts) == 1


class TestMemoryExtractorIntegration:
    """Independent integration tests for MemoryExtractor."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM gateway."""
        mock = AsyncMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.fixture
    def extractor(self, mock_llm):
        """Create MemoryExtractor with mock LLM."""
        return MemoryExtractor(llm_gateway=mock_llm, model_name="test-model")

    @pytest.mark.asyncio
    async def test_full_workflow(self, extractor, mock_llm):
        """Test complete workflow: extract then check importance."""
        # First, check if message is important
        msg = Message(role="user", content="I have a peanut allergy", sender_id="user1")
        mock_llm.generate.return_value = "true"

        is_important = await extractor.is_important(msg)
        assert is_important is True

        # Then extract facts
        messages = [
            Message(role="user", content="I have a peanut allergy", sender_id="user1"),
            Message(role="assistant", content="I'll remember that", sender_id="assistant")
        ]
        mock_llm.generate.return_value = '''
        [
            {
                "content": "User has a peanut allergy",
                "metadata": {"type": "medical", "severity": "high"},
                "user_id": "user1"
            }
        ]
        '''

        facts = await extractor.extract(messages)
        assert len(facts) == 1
        assert facts[0].content == "User has a peanut allergy"
        assert facts[0].metadata["type"] == "medical"

    @pytest.mark.asyncio
    async def test_extract_with_multiple_users(self, extractor, mock_llm):
        """Test extracting facts from multiple users."""
        messages = [
            Message(role="user", content="I'm Alice, I like cats", sender_id="alice"),
            Message(role="user", content="I'm Bob, I prefer dogs", sender_id="bob"),
            Message(role="assistant", content="I'll remember both", sender_id="assistant")
        ]

        mock_llm.generate.return_value = '''
        [
            {"content": "Alice likes cats", "metadata": {}, "user_id": "alice"},
            {"content": "Bob prefers dogs", "metadata": {}, "user_id": "bob"}
        ]
        '''

        facts = await extractor.extract(messages)
        assert len(facts) == 2
        user_ids = {f.user_id for f in facts}
        assert "alice" in user_ids
        assert "bob" in user_ids
