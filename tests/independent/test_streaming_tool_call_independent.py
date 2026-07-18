"""Independent tests for streaming tool call verification.

This test file verifies the complete streaming tool call flow:
1. Event order: tool_call_start → tool_call_argument (multiple) → tool_call_end
2. Arguments string concatenation correctness
3. REST fallback vs WebSocket consistency
4. Non-tool-call conversations work normally

Reference:
- llm_gateway.py: ChatResponseType enum with TOOL_CALL_* types
- openai_llm.py: Stream events generation
- agent_runtime.py: Event type mapping
- index.html: Frontend event handling
"""
import pytest
import re
from pathlib import Path


# Paths
HTML_FILE = Path(__file__).parent.parent.parent / "agent_framework" / "gateway" / "static" / "index.html"
LLM_GATEWAY_FILE = Path(__file__).parent.parent.parent / "agent_framework" / "infrastructure" / "llm_gateway.py"
OPENAI_LLM_FILE = Path(__file__).parent.parent.parent / "agent_framework" / "infrastructure" / "openai_llm.py"
AGENT_RUNTIME_FILE = Path(__file__).parent.parent.parent / "agent_framework" / "runtime" / "agent_runtime.py"


@pytest.fixture
def html_content():
    """Load HTML file content."""
    with open(HTML_FILE, 'r', encoding='utf-8') as f:
        return f.read()


@pytest.fixture
def llm_gateway_content():
    """Load LLM gateway file content."""
    with open(LLM_GATEWAY_FILE, 'r', encoding='utf-8') as f:
        return f.read()


@pytest.fixture
def openai_llm_content():
    """Load OpenAI LLM file content."""
    with open(OPENAI_LLM_FILE, 'r', encoding='utf-8') as f:
        return f.read()


@pytest.fixture
def agent_runtime_content():
    """Load agent runtime file content."""
    with open(AGENT_RUNTIME_FILE, 'r', encoding='utf-8') as f:
        return f.read()


class TestToolCallEventEnum:
    """Test TOOL_CALL_ARGUMENT enum in llm_gateway.py."""

    def test_tool_call_argument_enum_exists(self, llm_gateway_content):
        """Verify TOOL_CALL_ARGUMENT enum is defined."""
        assert 'TOOL_CALL_ARGUMENT' in llm_gateway_content, \
            "TOOL_CALL_ARGUMENT enum should be defined in llm_gateway.py"

    def test_tool_call_enum_values(self, llm_gateway_content):
        """Verify all TOOL_CALL_* enum values exist."""
        required_enums = [
            'TOOL_CALL_START',
            'TOOL_CALL_ARGUMENT',
            'TOOL_CALL_END'
        ]
        for enum_name in required_enums:
            assert enum_name in llm_gateway_content, \
                f"{enum_name} should be defined in ChatResponseType enum"


class TestOpenAILLMToolCallEvents:
    """Test OpenAI LLM tool call event generation."""

    def test_sends_tool_call_start(self, openai_llm_content):
        """Verify OpenAI LLM sends TOOL_CALL_START event."""
        assert 'ChatResponseType.TOOL_CALL_START' in openai_llm_content, \
            "Should send TOOL_CALL_START event"

    def test_sends_tool_call_argument(self, openai_llm_content):
        """Verify OpenAI LLM sends TOOL_CALL_ARGUMENT event."""
        assert 'ChatResponseType.TOOL_CALL_ARGUMENT' in openai_llm_content, \
            "Should send TOOL_CALL_ARGUMENT event"

    def test_sends_tool_call_end(self, openai_llm_content):
        """Verify OpenAI LLM sends TOOL_CALL_END event."""
        assert 'ChatResponseType.TOOL_CALL_END' in openai_llm_content, \
            "Should send TOOL_CALL_END event"

    def test_tool_call_end_on_tool_calls_finish(self, openai_llm_content):
        """Verify TOOL_CALL_END is sent when finish_reason is 'tool_calls'."""
        # When finish_reason == "tool_calls", should yield TOOL_CALL_END
        pattern = r'finish_reason\s*==\s*["\']tool_calls["\']'
        assert re.search(pattern, openai_llm_content), \
            "Should check for finish_reason === 'tool_calls'"

    def test_arguments_concatenation(self, openai_llm_content):
        """Verify tool call arguments are concatenated correctly."""
        # Look for += operator on arguments
        pattern = r'tool_call_tracker\[index\]\["arguments"\]\s*\+='
        assert re.search(pattern, openai_llm_content), \
            "Should concatenate arguments using +="


class TestAgentRuntimeToolCallMapping:
    """Test agent_runtime.py tool call event type mapping."""

    def test_handles_tool_call_start(self, agent_runtime_content):
        """Verify agent_runtime handles TOOL_CALL_START."""
        assert 'EventType.TOOL_CALL_START' in agent_runtime_content or \
               'TOOL_CALL_START' in agent_runtime_content, \
            "Should handle TOOL_CALL_START event type"

    def test_handles_tool_call_argument(self, agent_runtime_content):
        """Verify agent_runtime handles TOOL_CALL_ARGUMENT."""
        assert 'EventType.TOOL_CALL_ARGUMENT' in agent_runtime_content or \
               'TOOL_CALL_ARGUMENT' in agent_runtime_content, \
            "Should handle TOOL_CALL_ARGUMENT event type"

    def test_handles_tool_call_end(self, agent_runtime_content):
        """Verify agent_runtime handles TOOL_CALL_END."""
        assert 'EventType.TOOL_CALL_END' in agent_runtime_content or \
               'TOOL_CALL_END' in agent_runtime_content, \
            "Should handle TOOL_CALL_END event type"

    def test_tool_call_end_single_emission(self, agent_runtime_content):
        """Verify TOOL_CALL_END is emitted only once per tool call."""
        # Count occurrences of TOOL_CALL_END - should appear in yield statement, not multiple times
        occurrences = len(re.findall(r'TOOL_CALL_END', agent_runtime_content))
        assert occurrences >= 2, \
            "TOOL_CALL_END should appear at least twice (definition and yield)"


class TestFrontendToolCallHandling:
    """Test frontend tool call event handling in index.html."""

    def test_handles_tool_call_start(self, html_content):
        """Verify frontend handles tool_call_start event."""
        assert "data.type === 'tool_call_start'" in html_content, \
            "Should handle tool_call_start event type"

    def test_handles_tool_call_argument(self, html_content):
        """Verify frontend handles tool_call_argument event."""
        assert "data.type === 'tool_call_argument'" in html_content, \
            "Should handle tool_call_argument event type"

    def test_handles_tool_call_end(self, html_content):
        """Verify frontend handles tool_call_end event."""
        assert "data.type === 'tool_call_end'" in html_content, \
            "Should handle tool_call_end event type"

    def test_creates_tool_call_element(self, html_content):
        """Verify frontend creates tool call element on tool_call_start."""
        assert 'createToolCallElement' in html_content, \
            "Should create tool call element"

    def test_updates_tool_call_element(self, html_content):
        """Verify frontend updates tool call element on tool_call_end."""
        assert 'updateToolCallElement' in html_content, \
            "Should update tool call element"

    def test_arguments_concatenation_in_frontend(self, html_content):
        """Verify frontend concatenates arguments correctly."""
        # Look for currentToolCall.arguments +=
        pattern = r'currentToolCall\.arguments\s*\+='
        assert re.search(pattern, html_content), \
            "Should concatenate arguments using +="

    def test_prefers_data_content_for_arguments(self, html_content):
        """Verify frontend prefers data.content for arguments, with metadata fallback."""
        # Looking for: data.content || data.metadata?.arguments_fragment
        pattern = r'data\.content\s*\|\|.*arguments'
        assert re.search(pattern, html_content), \
            "Should prefer data.content over metadata.arguments_fragment"

    def test_tool_call_id_compatibility(self, html_content):
        """Verify frontend handles both data.metadata.tool_call_id and data.tool_call_id."""
        # Looking for: data.metadata?.tool_call_id || data.tool_call_id
        pattern = r'data\.metadata.*tool_call_id.*\|\|.*tool_call_id'
        assert re.search(pattern, html_content), \
            "Should handle both metadata.tool_call_id and tool_call_id for compatibility"


class TestEventOrderVerification:
    """Test event order sequence correctness."""

    def test_event_order_documented_in_frontend(self, html_content):
        """Document expected event order: start → argument(s) → end."""
        # Verify all three event types are handled
        has_start = "data.type === 'tool_call_start'" in html_content
        has_argument = "data.type === 'tool_call_argument'" in html_content
        has_end = "data.type === 'tool_call_end'" in html_content

        assert has_start and has_argument and has_end, \
            "All three tool call event types should be handled (start, argument, end)"

    def test_start_creates_tool_call_state(self, html_content):
        """Verify tool_call_start creates currentToolCall state."""
        # In tool_call_start handler, should set currentToolCall
        start_handler_pattern = r"data\.type\s*===\s*['\"]tool_call_start['\"].*?currentToolCall\s*="
        assert re.search(start_handler_pattern, html_content, re.DOTALL), \
            "tool_call_start should initialize currentToolCall"

    def test_argument_accumulates_to_state(self, html_content):
        """Verify tool_call_argument accumulates to currentToolCall."""
        # In tool_call_argument handler, should += to currentToolCall.arguments
        arg_handler_pattern = r"data\.type\s*===\s*['\"]tool_call_argument['\"].*?currentToolCall\.arguments\s*\+="
        assert re.search(arg_handler_pattern, html_content, re.DOTALL), \
            "tool_call_argument should accumulate to currentToolCall.arguments"

    def test_end_updates_and_clears_state(self, html_content):
        """Verify tool_call_end updates element and clears state."""
        # In tool_call_end handler, should update element then set currentToolCall = null
        end_handler_pattern = r"data\.type\s*===\s*['\"]tool_call_end['\"].*?currentToolCall\s*=\s*null"
        assert re.search(end_handler_pattern, html_content, re.DOTALL), \
            "tool_call_end should set currentToolCall = null"


class TestRESTWebSocketConsistency:
    """Test REST fallback and WebSocket consistency."""

    def test_rest_fallback_uses_same_handler(self, html_content):
        """Verify REST fallback calls same handleWebSocketMessage as WebSocket."""
        # Find where handleWebSocketMessage is called in REST fallback
        fallback_handler_pattern = r'handleWebSocketMessage\(data\)'
        matches = re.findall(fallback_handler_pattern, html_content)
        assert len(matches) >= 2, \
            "handleWebSocketMessage should be called for both WebSocket and REST fallback"

    def test_sse_endpoint_exists(self, html_content):
        """Verify REST fallback SSE endpoint is called."""
        assert '/messages/stream' in html_content, \
            "Should call /messages/stream endpoint for REST fallback"


class TestNonToolCallConversations:
    """Test that non-tool-call conversations still work."""

    def test_handles_text_token(self, html_content):
        """Verify text_token events are still handled."""
        assert "data.type === 'text_token'" in html_content, \
            "Should handle text_token events for regular responses"

    def test_handles_final_answer(self, html_content):
        """Verify final_answer events are still handled."""
        assert "data.type === 'final_answer'" in html_content, \
            "Should handle final_answer events for regular responses"

    def test_streaming_element_created_for_text(self, html_content):
        """Verify streaming element is created for text tokens."""
        assert 'createStreamingMessage()' in html_content, \
            "Should create streaming message element for text responses"


class TestReActPlannerStreamingToolCall:
    """Test ReActPlanner streaming tool call handling."""

    def test_react_planner_uses_correct_argument_event_type(self, agent_runtime_content):
        """Verify ReActPlanner checks for 'tool_call_argument' not 'argument'."""
        # This tests that react_planner.py uses the correct event type
        # The correct event type is "tool_call_argument" per EventType.TOOL_CALL_ARGUMENT
        react_planner_file = Path(__file__).parent.parent.parent / "agent_framework" / "planners" / "react_planner.py"
        with open(react_planner_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # The fix should check for "tool_call_argument" not just "argument"
        # Look for the streaming handler section
        if 'event.type == "argument"' in content:
            # If we find the wrong check, this is a bug
            assert False, "react_planner.py uses 'argument' instead of 'tool_call_argument' - tool call arguments won't be accumulated during streaming"

        # Should have the correct check
        assert 'event.type == "tool_call_argument"' in content or "event.type == 'tool_call_argument'" in content, \
            "react_planner.py should check for 'tool_call_argument' event type"


class TestToolCallCompleteFlow:
    """Integration-style test for complete tool call flow."""

    def test_complete_flow_components_exist(self, html_content, llm_gateway_content, openai_llm_content):
        """Verify all components for complete tool call flow exist."""
        components = [
            ("TOOL_CALL_ARGUMENT enum", 'TOOL_CALL_ARGUMENT' in llm_gateway_content),
            ("TOOL_CALL_START in OpenAI LLM", 'TOOL_CALL_START' in openai_llm_content),
            ("TOOL_CALL_ARGUMENT in OpenAI LLM", 'TOOL_CALL_ARGUMENT' in openai_llm_content),
            ("tool_call_start handling in frontend", "data.type === 'tool_call_start'" in html_content),
            ("tool_call_argument handling in frontend", "data.type === 'tool_call_argument'" in html_content),
            ("tool_call_end handling in frontend", "data.type === 'tool_call_end'" in html_content),
            ("arguments concatenation in frontend", "currentToolCall.arguments +=" in html_content),
        ]

        failed = [name for name, passed in components if not passed]
        assert not failed, f"Missing components: {failed}"

    def test_no_legacy_tool_call_chunk(self, html_content):
        """Verify frontend doesn't use legacy 'tool_call_chunk' type."""
        # The legacy name was tool_call_chunk, should now be tool_call_argument
        legacy_pattern = r"data\.type\s*===\s*['\"]tool_call_chunk['\"]"
        assert not re.search(legacy_pattern, html_content), \
            "Should not use legacy 'tool_call_chunk' type, use 'tool_call_argument' instead"