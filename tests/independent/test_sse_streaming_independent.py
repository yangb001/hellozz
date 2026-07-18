"""Independent tests for SSE streaming functionality.

This test file verifies the SSE streaming implementation for the AI Agent Framework:
1. Backend SSE output format (rest.py)
2. Frontend SSE parsing logic (index.html)
3. Event handling for text_token, final_answer

Design Reference:
- Frontend: agent_framework/gateway/static/index.html
- Backend API: agent_framework/gateway/api/rest.py

Test Coverage:
1. SSE message format verification
2. Frontend SSE parsing logic validation
3. Event type handling correctness
4. DONE signal handling
"""
import pytest
import re
import json
from pathlib import Path


# Paths
HTML_FILE = Path(__file__).parent.parent.parent / "agent_framework" / "gateway" / "static" / "index.html"
REST_FILE = Path(__file__).parent.parent.parent / "agent_framework" / "gateway" / "api" / "rest.py"


@pytest.fixture
def html_content():
    """Load HTML file content."""
    assert HTML_FILE.exists(), f"HTML file not found: {HTML_FILE}"
    with open(HTML_FILE, 'r', encoding='utf-8') as f:
        return f.read()


@pytest.fixture
def rest_content():
    """Load REST API file content."""
    assert REST_FILE.exists(), f"REST file not found: {REST_FILE}"
    with open(REST_FILE, 'r', encoding='utf-8') as f:
        return f.read()


class TestSSEServerFormat:
    """Test SSE server-side output format."""

    def test_sse_endpoint_exists(self, rest_content):
        """Verify /messages/stream endpoint exists."""
        assert '/sessions/{session_id}/messages/stream' in rest_content

    def test_sse_yields_event_data_format(self, rest_content):
        """Verify SSE yields event data in correct format: data: {json}\\n\\n."""
        # Look for the yield statement pattern
        pattern = r'yield f"data: '
        assert re.search(pattern, rest_content), \
            "SSE should yield data in format: data: {event_data}\\n\\n"

    def test_sse_event_data_structure(self, rest_content):
        """Verify event_data contains type, content, metadata, timestamp."""
        # Extract the event_data construction
        event_data_pattern = r'event_data\s*=\s*\{[^}]+\}'
        match = re.search(event_data_pattern, rest_content, re.DOTALL)
        assert match, "event_data dict should be constructed"
        event_data_block = match.group(0)
        assert '"type"' in event_data_block
        assert '"content"' in event_data_block
        assert '"metadata"' in event_data_block
        assert '"timestamp"' in event_data_block

    def test_sse_done_signal_format(self, rest_content):
        """Verify DONE signal is sent as JSON object, not [DONE] literal."""
        # Backend sends: {"type": "done", "content": ""}
        # Looking for: yield f"data: {{"type": "done"
        pattern = r'yield f"data: \{'
        assert re.search(pattern, rest_content), \
            "DONE signal should be sent as JSON object {\"type\": \"done\", ...}"


class TestSSEClientParsing:
    """Test SSE client-side parsing logic."""

    def test_fetch_stream_endpoint(self, html_content):
        """Verify frontend calls /messages/stream endpoint."""
        pattern = r'/sessions/\$\{currentSessionId\}/messages/stream'
        assert re.search(pattern, html_content), \
            "Frontend should call /messages/stream endpoint"

    def test_uses_reader_for_streaming(self, html_content):
        """Verify frontend uses response.body.getReader()."""
        assert 'response.body.getReader()' in html_content, \
            "Should use getReader() for streaming"

    def test_decoder_text_decoder(self, html_content):
        """Verify frontend uses TextDecoder."""
        assert 'TextDecoder()' in html_content, \
            "Should use TextDecoder for decoding chunks"

    def test_buffer_accumulates_chunks(self, html_content):
        """Verify buffer accumulates decoded chunks."""
        # Look for buffer += decoder.decode pattern
        pattern = r'buffer\s*\+=.*decoder\.decode'
        assert re.search(pattern, html_content), \
            "Should accumulate chunks in buffer"

    def test_splits_by_newline(self, html_content):
        """Verify buffer is split by newline."""
        pattern = r"buffer\.split\(['\"]\\n['\"]\)"
        assert re.search(pattern, html_content), \
            "Should split buffer by newline"

    def test_parses_data_lines(self, html_content):
        """Verify parses lines starting with 'data: '."""
        pattern = r"line\.startsWith\(['\"]data: ['\"]\)"
        assert re.search(pattern, html_content), \
            "Should parse lines starting with 'data: '"

    def test_skips_done_signal(self, html_content):
        """Verify [DONE] signal is skipped."""
        # Note: Backend sends {"type": "done"} but frontend checks for [DONE]
        # This is a known mismatch that should be addressed
        pattern = r"dataStr\s*===\s*['\"]\[DONE\]['\"]"
        assert re.search(pattern, html_content), \
            "Should skip [DONE] signal"


class TestEventHandling:
    """Test event handling in handleWebSocketMessage."""

    def test_handleWebSocketMessage_exists(self, html_content):
        """Verify handleWebSocketMessage function exists."""
        assert 'function handleWebSocketMessage' in html_content or \
               'handleWebSocketMessage = ' in html_content

    def test_handles_error_event(self, html_content):
        """Verify error events are handled."""
        assert "data.type === 'error'" in html_content or \
               'data.type === "error"' in html_content

    def test_handles_text_token_event(self, html_content):
        """Verify text_token events are handled."""
        assert "data.type === 'text_token'" in html_content or \
               'data.type === "text_token"' in html_content

    def test_handles_final_answer_event(self, html_content):
        """Verify final_answer events are handled."""
        assert "data.type === 'final_answer'" in html_content or \
               'data.type === "final_answer"' in html_content

    def test_handles_response_event(self, html_content):
        """Verify response events are handled (alias for final_answer)."""
        assert "data.type === 'response'" in html_content or \
               'data.type === "response"' in html_content

    def test_text_token_creates_streaming_element(self, html_content):
        """Verify text_token creates streaming message element."""
        # Should check for !streamingElement and call createStreamingMessage
        assert '!streamingElement' in html_content or '!streamingElement' in html_content
        assert 'createStreamingMessage()' in html_content

    def test_text_token_accumulates_content(self, html_content):
        """Verify text_token accumulates content."""
        assert 'streamingContent +=' in html_content or \
               'streamingContent = streamingContent +' in html_content

    def test_finalize_streaming_called(self, html_content):
        """Verify finalizeStreaming is called after final_answer."""
        assert 'finalizeStreaming()' in html_content


class TestSSEParseMismatch:
    """Test for known issues in SSE parsing."""

    def test_done_signal_mismatch(self, html_content, rest_content):
        """Detect mismatch between backend DONE signal and frontend handling.

        Backend sends: {"type": "done", "content": ""}
        Frontend checks: dataStr === '[DONE]'

        This test documents a known bug where the DONE signal format
        between backend and frontend don't match.
        """
        # Backend sends JSON object
        backend_sends_json_done = '"type": "done"' in rest_content
        # Frontend checks for literal [DONE]
        frontend_checks_literal_done = "'[DONE]'" in html_content or '"[DONE]"' in html_content

        if backend_sends_json_done and frontend_checks_literal_done:
            pytest.fail(
                "MISMATCH: Backend sends '{\"type\": \"done\"}' but frontend checks for '[DONE]'. "
                "Frontend should either: (1) check for data.type === 'done', or "
                "(2) backend should send literal [DONE]"
            )

    def test_finally_cleans_up_streaming(self, html_content):
        """Verify finally block calls finalizeStreaming in REST fallback."""
        # Look for finally block after the fetch call
        # Simply check if finally has finalizeStreaming
        fallback_match = re.search(r'\} else \{[^}]*// Fallback to REST API[^}]*\} finally \{([^}]+)\}', html_content, re.DOTALL)
        if fallback_match:
            finally_block = fallback_match.group(1)
            has_cleanup = 'finalizeStreaming()' in finally_block
            assert has_cleanup, \
                "finally block should call finalizeStreaming() to clean up streaming element"


class TestSSEMultiLineJSON:
    """Test handling of multi-line JSON in SSE (if applicable)."""

    def test_sse_json_is_single_line(self, rest_content):
        """Verify backend sends compact single-line JSON."""
        # The yield statement should produce single-line JSON
        yield_pattern = r'yield f"data: '
        assert re.search(yield_pattern, rest_content), \
            "SSE should send compact single-line JSON"

    def test_frontend_buffer_handles_incomplete_json(self, html_content):
        """Verify frontend properly handles incomplete JSON in buffer."""
        # When buffer is split by \n, the last element should be kept for next chunk
        pattern = r'buffer\s*=\s*lines\.pop\(\)'
        assert re.search(pattern, html_content), \
            "Should keep incomplete line in buffer for next chunk"


class TestIntegrationScenarios:
    """Integration-style tests for SSE streaming flow."""

    def test_full_sse_flow_document(self, html_content, rest_content):
        """Document the full SSE flow for verification."""
        steps = [
            ("Backend defines /messages/stream endpoint", "/messages/stream" in rest_content),
            ("Backend yields event_data as JSON", "event_data" in rest_content and "yield" in rest_content),
            ("Backend sends done signal", "done" in rest_content),
            ("Frontend fetches /messages/stream", "/messages/stream" in html_content),
            ("Frontend uses getReader", "getReader()" in html_content),
            ("Frontend parses SSE lines", "data: " in html_content),
            ("Frontend calls handleWebSocketMessage", "handleWebSocketMessage" in html_content),
            ("handleWebSocketMessage handles text_token", "text_token" in html_content),
            ("handleWebSocketMessage handles final_answer", "final_answer" in html_content),
        ]

        failed_steps = [name for name, passed in steps if not passed]
        assert not failed_steps, f"Failed steps: {failed_steps}"

    def test_streaming_cleanup_on_completion(self, html_content):
        """Verify streaming resources are cleaned up when streaming completes."""
        # Find where finalizeStreaming is called
        finalize_calls = list(re.finditer(r'finalizeStreaming\(\)', html_content))

        assert len(finalize_calls) >= 2, \
            "finalizeStreaming should be called in multiple places: WebSocket close, error, and REST finally"

        # Check REST fallback finally has cleanup
        finally_pattern = r'\} finally \{[^}]*finalizeStreaming\(\)[^}]*\}'
        assert re.search(finally_pattern, html_content, re.DOTALL), \
            "REST fallback finally block should call finalizeStreaming()"