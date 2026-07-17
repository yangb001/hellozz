"""Independent tests for Web page bug fixes verification.

This test file verifies 3 bug fixes in the multi-session web page:

Bug 1: Streaming response not displayed
- Added text_token event handling
- New streaming display logic (streamingElement, streamingContent)

Bug 2: Send button stuck showing "Processing"
- Reset isProcessing state on session switch
- Reset isProcessing state on WebSocket close/error

Bug 3: Scroll issue
- Fixed chat box internal scrolling with overflow: hidden and min-height: 0

Test Coverage:
1. Streaming response display
2. Send button state reset
3. Chat box internal scrolling
4. Code quality review
"""
import pytest
import re
from pathlib import Path


HTML_FILE = Path(__file__).parent.parent.parent / "agent_framework" / "gateway" / "static" / "index.html"


@pytest.fixture
def html_content():
    """Load the HTML file content."""
    assert HTML_FILE.exists(), f"HTML file not found: {HTML_FILE}"
    with open(HTML_FILE, 'r', encoding='utf-8') as f:
        return f.read()


class TestStreamingResponse:
    """Test Bug 1 fix: Streaming response display with text_token events."""

    def test_text_token_handler_exists(self, html_content):
        """Must handle text_token event type."""
        assert "data.type === 'text_token'" in html_content or \
               'data.type === "text_token"' in html_content, \
               "Missing text_token event handler"

    def test_streaming_state_variables(self, html_content):
        """Must have streaming state variables."""
        assert 'streamingElement' in html_content, "Missing streamingElement variable"
        assert 'streamingContent' in html_content, "Missing streamingContent variable"

    def test_create_streaming_message_function(self, html_content):
        """Must have function to create streaming message element."""
        assert 'function createStreamingMessage()' in html_content, \
               "Missing createStreamingMessage() function"

    def test_update_streaming_content_function(self, html_content):
        """Must have function to update streaming content."""
        assert 'function updateStreamingContent(' in html_content, \
               "Missing updateStreamingContent() function"

    def test_finalize_streaming_function(self, html_content):
        """Must have function to finalize streaming message."""
        assert 'function finalizeStreaming()' in html_content, \
               "Missing finalizeStreaming() function"

    def test_streaming_creates_message_element(self, html_content):
        """createStreamingMessage should create a message div."""
        func_match = re.search(
            r'function createStreamingMessage\(\)\s*\{(.+?)\n\s*\}',
            html_content,
            re.DOTALL
        )
        assert func_match, "Could not find createStreamingMessage function body"
        func_body = func_match.group(1)
        assert "createElement('div')" in func_body or 'createElement("div")' in func_body, \
            "Should create a div element"
        assert "message assistant" in func_body, "Should have assistant message class"

    def test_streaming_has_id(self, html_content):
        """Streaming element should have id for tracking."""
        assert "streamingMessage" in html_content, "Missing streamingMessage element id"
        assert "streamingBubble" in html_content, "Missing streamingBubble element id"

    def test_streaming_accumulates_content(self, html_content):
        """Should accumulate streaming content."""
        # Look for += pattern on streamingContent
        assert "streamingContent += data.content" in html_content or \
               "streamingContent += " in html_content, \
               "Should accumulate streaming content"

    def test_streaming_updates_bubble(self, html_content):
        """Should update bubble with accumulated content."""
        func_match = re.search(
            r'function updateStreamingContent\(.+?\)\s*\{(.+?)\n\s*\}',
            html_content,
            re.DOTALL
        )
        assert func_match, "Could not find updateStreamingContent function body"
        func_body = func_match.group(1)
        assert "textContent" in func_body or "innerHTML" in func_body, \
            "Should update element content"

    def test_streaming_uses_text_content_not_innerhtml(self, html_content):
        """Streaming should use textContent for safety (XSS prevention)."""
        func_match = re.search(
            r'function updateStreamingContent\(.+?\)\s*\{(.+?)\n\s*\}',
            html_content,
            re.DOTALL
        )
        assert func_match, "Could not find updateStreamingContent function body"
        func_body = func_match.group(1)
        assert "textContent" in func_body, \
            "Should use textContent instead of innerHTML for XSS prevention"

    def test_streaming_removes_on_finalize(self, html_content):
        """finalizeStreaming should remove the streaming element."""
        func_match = re.search(
            r'function finalizeStreaming\(\).*?\n(.*?)(?=\n        function |\Z)',
            html_content,
            re.DOTALL
        )
        assert func_match, "Could not find finalizeStreaming function body"
        func_body = func_match.group(1)
        assert ".remove()" in func_body, "Should remove streaming element"
        assert "streamingElement = null" in func_body, "Should reset streamingElement to null"
        assert "streamingContent = ''" in func_body, "Should reset streamingContent"

    def test_streaming_hides_typing_indicator(self, html_content):
        """Should hide typing indicator when streaming starts."""
        # In handleWebSocketMessage, text_token handler should hide typing
        handler_match = re.search(
            r"if \(data\.type === 'text_token'\)\s*\{(.+?)\n\s*\}",
            html_content,
            re.DOTALL
        )
        assert handler_match, "Could not find text_token handler"
        handler_body = handler_match.group(1)
        assert "hideTypingIndicator" in handler_body, \
            "Should hide typing indicator when streaming starts"

    def test_finalize_called_before_final_answer(self, html_content):
        """Should finalize streaming before appending final answer."""
        handler_match = re.search(
            r"if \(data\.type === 'final_answer'.+?\)\s*\{(.+?)\n\s*\}",
            html_content,
            re.DOTALL
        )
        assert handler_match, "Could not find final_answer handler"
        handler_body = handler_match.group(1)
        assert "finalizeStreaming" in handler_body, \
            "Should finalize streaming before final answer"

    def test_finalize_called_on_error(self, html_content):
        """Should finalize streaming on error."""
        error_match = re.search(
            r"if \(data\.type === 'error'\)\s*\{(.+?)\n\s*\}",
            html_content,
            re.DOTALL
        )
        assert error_match, "Could not find error handler"
        error_body = error_match.group(1)
        assert "finalizeStreaming" in error_body, \
            "Should finalize streaming on error"

    def test_streaming_scroll_to_bottom(self, html_content):
        """Should scroll to bottom during streaming."""
        func_match = re.search(
            r'function updateStreamingContent\(.+?\)\s*\{(.+?)\n\s*\}',
            html_content,
            re.DOTALL
        )
        assert func_match, "Could not find updateStreamingContent function body"
        func_body = func_match.group(1)
        assert "scrollToBottom" in func_body, "Should scroll during streaming"


class TestSendButtonStateReset:
    """Test Bug 2 fix: Send button state reset."""

    def test_select_session_resets_processing(self, html_content):
        """selectSession should reset isProcessing state."""
        func_match = re.search(
            r'async function selectSession\(.+?\)\s*\{(.+?)\n\s*\}',
            html_content,
            re.DOTALL
        )
        assert func_match, "Could not find selectSession function body"
        func_body = func_match.group(1)
        assert "isProcessing = false" in func_body, \
            "Should reset isProcessing on session switch"
        assert "updateSendButton" in func_body, \
            "Should update send button on session switch"

    def test_select_session_hides_typing_indicator(self, html_content):
        """selectSession should hide typing indicator."""
        func_match = re.search(
            r'async function selectSession\(.+?\)\s*\{(.+?)\n\s*\}',
            html_content,
            re.DOTALL
        )
        assert func_match, "Could not find selectSession function body"
        func_body = func_match.group(1)
        assert "hideTypingIndicator" in func_body, \
            "Should hide typing indicator on session switch"

    def test_select_session_finalizes_streaming(self, html_content):
        """selectSession should finalize any active streaming."""
        func_match = re.search(
            r'async function selectSession\(.+?\)\s*\{(.+?)\n\s*\}',
            html_content,
            re.DOTALL
        )
        assert func_match, "Could not find selectSession function body"
        func_body = func_match.group(1)
        assert "finalizeStreaming" in func_body, \
            "Should finalize streaming on session switch"

    def test_ws_onclose_resets_processing(self, html_content):
        """WebSocket onclose should reset isProcessing."""
        onclose_match = re.search(
            r'ws\.onclose\s*=\s*\(\)\s*=>\s*\{(.+?)\n\s*\};',
            html_content,
            re.DOTALL
        )
        assert onclose_match, "Could not find ws.onclose handler"
        onclose_body = onclose_match.group(1)
        assert "isProcessing = false" in onclose_body, \
            "Should reset isProcessing on WebSocket close"
        assert "updateSendButton" in onclose_body, \
            "Should update send button on WebSocket close"

    def test_ws_onclose_hides_typing(self, html_content):
        """WebSocket onclose should hide typing indicator."""
        onclose_match = re.search(
            r'ws\.onclose\s*=\s*\(\)\s*=>\s*\{(.+?)\n\s*\};',
            html_content,
            re.DOTALL
        )
        assert onclose_match, "Could not find ws.onclose handler"
        onclose_body = onclose_match.group(1)
        assert "hideTypingIndicator" in onclose_body, \
            "Should hide typing indicator on WebSocket close"

    def test_ws_onclose_finalizes_streaming(self, html_content):
        """WebSocket onclose should finalize streaming."""
        onclose_match = re.search(
            r'ws\.onclose\s*=\s*\(\)\s*=>\s*\{(.+?)\n\s*\};',
            html_content,
            re.DOTALL
        )
        assert onclose_match, "Could not find ws.onclose handler"
        onclose_body = onclose_match.group(1)
        assert "finalizeStreaming" in onclose_body, \
            "Should finalize streaming on WebSocket close"

    def test_ws_onerror_resets_processing(self, html_content):
        """WebSocket onerror should reset isProcessing."""
        onerror_match = re.search(
            r'ws\.onerror\s*=\s*\(.+?\)\s*=>\s*\{(.+?)\n\s*\};',
            html_content,
            re.DOTALL
        )
        assert onerror_match, "Could not find ws.onerror handler"
        onerror_body = onerror_match.group(1)
        assert "isProcessing = false" in onerror_body, \
            "Should reset isProcessing on WebSocket error"
        assert "updateSendButton" in onerror_body, \
            "Should update send button on WebSocket error"

    def test_ws_onerror_hides_typing(self, html_content):
        """WebSocket onerror should hide typing indicator."""
        onerror_match = re.search(
            r'ws\.onerror\s*=\s*\(.+?\)\s*=>\s*\{(.+?)\n\s*\};',
            html_content,
            re.DOTALL
        )
        assert onerror_match, "Could not find ws.onerror handler"
        onerror_body = onerror_match.group(1)
        assert "hideTypingIndicator" in onerror_body, \
            "Should hide typing indicator on WebSocket error"

    def test_ws_onerror_finalizes_streaming(self, html_content):
        """WebSocket onerror should finalize streaming."""
        onerror_match = re.search(
            r'ws\.onerror\s*=\s*\(.+?\)\s*=>\s*\{(.+?)\n\s*\};',
            html_content,
            re.DOTALL
        )
        assert onerror_match, "Could not find ws.onerror handler"
        onerror_body = onerror_match.group(1)
        assert "finalizeStreaming" in onerror_body, \
            "Should finalize streaming on WebSocket error"

    def test_send_button_disabled_state(self, html_content):
        """Send button should have disabled state styling."""
        assert 'send-btn:disabled' in html_content or \
               '.send-btn:disabled' in html_content, \
               "Should have disabled button styling"

    def test_update_send_button_function(self, html_content):
        """updateSendButton should toggle button text and disabled state."""
        func_match = re.search(
            r'function updateSendButton\(\)\s*\{(.+?)\n\s*\}',
            html_content,
            re.DOTALL
        )
        assert func_match, "Could not find updateSendButton function body"
        func_body = func_match.group(1)
        assert "disabled" in func_body, "Should set disabled attribute"
        assert "处理中" in func_body, "Should show 'Processing' text"
        assert "发送" in func_body, "Should show 'Send' text"


class TestScrollFix:
    """Test Bug 3 fix: Chat box internal scrolling."""

    def test_main_overflow_hidden(self, html_content):
        """Main container should have overflow: hidden."""
        main_match = re.search(r'\.main\s*\{([^}]+)\}', html_content, re.DOTALL)
        assert main_match, "Could not find .main CSS"
        main_css = main_match.group(1)
        assert "overflow: hidden" in main_css, \
            "Main container should have overflow: hidden"

    def test_main_min_width_zero(self, html_content):
        """Main container should have min-width: 0 for flex containment."""
        main_match = re.search(r'\.main\s*\{([^}]+)\}', html_content, re.DOTALL)
        assert main_match, "Could not find .main CSS"
        main_css = main_match.group(1)
        assert "min-width: 0" in main_css, \
            "Main container should have min-width: 0"

    def test_chat_content_overflow_hidden(self, html_content):
        """#chatContent should have overflow: hidden."""
        chat_match = re.search(r'#chatContent\s*\{([^}]+)\}', html_content, re.DOTALL)
        assert chat_match, "Could not find #chatContent CSS"
        chat_css = chat_match.group(1)
        assert "overflow: hidden" in chat_css, \
            "#chatContent should have overflow: hidden"

    def test_chat_content_min_height_zero(self, html_content):
        """#chatContent should have min-height: 0 for flex containment."""
        chat_match = re.search(r'#chatContent\s*\{([^}]+)\}', html_content, re.DOTALL)
        assert chat_match, "Could not find #chatContent CSS"
        chat_css = chat_match.group(1)
        assert "min-height: 0" in chat_css, \
            "#chatContent should have min-height: 0"

    def test_chat_content_flex_column(self, html_content):
        """#chatContent should use flex column layout."""
        chat_match = re.search(r'#chatContent\s*\{([^}]+)\}', html_content, re.DOTALL)
        assert chat_match, "Could not find #chatContent CSS"
        chat_css = chat_match.group(1)
        assert "display: flex" in chat_css, "#chatContent should be flex"
        assert "flex-direction: column" in chat_css, \
            "#chatContent should use column direction"

    def test_chat_messages_overflow_auto(self, html_content):
        """chat-messages should have overflow-y: auto for scrolling."""
        messages_match = re.search(r'\.chat-messages\s*\{([^}]+)\}', html_content, re.DOTALL)
        assert messages_match, "Could not find .chat-messages CSS"
        messages_css = messages_match.group(1)
        assert "overflow-y: auto" in messages_css, \
            ".chat-messages should have overflow-y: auto"

    def test_chat_messages_min_height_zero(self, html_content):
        """chat-messages should have min-height: 0 for flex scroll."""
        messages_match = re.search(r'\.chat-messages\s*\{([^}]+)\}', html_content, re.DOTALL)
        assert messages_match, "Could not find .chat-messages CSS"
        messages_css = messages_match.group(1)
        assert "min-height: 0" in messages_css, \
            ".chat-messages should have min-height: 0"

    def test_empty_state_min_height_zero(self, html_content):
        """Empty state should have min-height: 0."""
        empty_match = re.search(r'\.empty-state\s*\{([^}]+)\}', html_content, re.DOTALL)
        assert empty_match, "Could not find .empty-state CSS"
        empty_css = empty_match.group(1)
        assert "min-height: 0" in empty_css, \
            ".empty-state should have min-height: 0"

    def test_scroll_to_bottom_targets_message_list(self, html_content):
        """scrollToBottom should scroll messageList, not the whole page."""
        func_match = re.search(
            r'function scrollToBottom\(\)\s*\{(.+?)\n\s*\}',
            html_content,
            re.DOTALL
        )
        assert func_match, "Could not find scrollToBottom function body"
        func_body = func_match.group(1)
        assert "messageList" in func_body, \
            "Should scroll messageList element"
        assert "scrollTop" in func_body, \
            "Should set scrollTop property"

    def test_height_100vh_on_body(self, html_content):
        """Body should have height: 100vh for full viewport."""
        body_match = re.search(r'body\s*\{([^}]+)\}', html_content, re.DOTALL)
        assert body_match, "Could not find body CSS"
        body_css = body_match.group(1)
        assert "height: 100vh" in body_css, "Body should have height: 100vh"


class TestCodeQualityReview:
    """Review code quality of all 3 fixes."""

    def test_streaming_state_initialized(self, html_content):
        """Streaming state variables should be properly initialized."""
        assert "let streamingElement = null" in html_content, \
            "streamingElement should be initialized to null"
        assert "let streamingContent = ''" in html_content or \
               'let streamingContent = ""' in html_content, \
               "streamingContent should be initialized to empty string"

    def test_no_memory_leak_on_finalize(self, html_content):
        """finalizeStreaming should clean up references."""
        func_match = re.search(
            r'function finalizeStreaming\(\).*?\n(.*?)(?=\n        function |\Z)',
            html_content,
            re.DOTALL
        )
        assert func_match, "Could not find finalizeStreaming function body"
        func_body = func_match.group(1)
        assert "streamingElement = null" in func_body, "Should null out reference"
        assert "streamingContent = ''" in func_body, "Should clear content"

    def test_consistent_state_reset_pattern(self, html_content):
        """All reset locations should follow same pattern: isProcessing + updateSendButton + hideTyping + finalize."""
        # Check selectSession
        select_match = re.search(
            r'async function selectSession\(.+?\)\s*\{(.+?)\n\s*\}',
            html_content,
            re.DOTALL
        )
        assert select_match
        select_body = select_match.group(1)

        # Check ws.onclose
        onclose_match = re.search(
            r'ws\.onclose\s*=\s*\(\)\s*=>\s*\{(.+?)\n\s*\};',
            html_content,
            re.DOTALL
        )
        assert onclose_match
        onclose_body = onclose_match.group(1)

        # Both should have the same reset pattern
        for body, name in [(select_body, "selectSession"), (onclose_body, "ws.onclose")]:
            assert "isProcessing = false" in body, f"{name} should reset isProcessing"
            assert "updateSendButton" in body, f"{name} should call updateSendButton"
            assert "hideTypingIndicator" in body, f"{name} should call hideTypingIndicator"
            assert "finalizeStreaming" in body, f"{name} should call finalizeStreaming"

    def test_streaming_null_check_before_use(self, html_content):
        """Should check streamingElement before using it."""
        # updateStreamingContent should check for null
        func_match = re.search(
            r'function updateStreamingContent\(.+?\)\s*\{(.+?)\n\s*\}',
            html_content,
            re.DOTALL
        )
        assert func_match
        func_body = func_match.group(1)
        assert "if (!element)" in func_body or "if (element)" in func_body, \
            "Should check element before use"

    def test_streaming_null_check_in_finalize(self, html_content):
        """finalizeStreaming should check if streamingElement exists."""
        func_match = re.search(
            r'function finalizeStreaming\(\).*?\n(.*?)(?=\n        function |\Z)',
            html_content,
            re.DOTALL
        )
        assert func_match
        func_body = func_match.group(1)
        assert "if (streamingElement)" in func_body, \
            "Should check streamingElement before cleanup"

    def test_css_scrollbar_styling(self, html_content):
        """Should have custom scrollbar styling for better UX."""
        assert "::-webkit-scrollbar" in html_content, "Should have custom scrollbar"
        assert "::-webkit-scrollbar-thumb" in html_content, "Should style scrollbar thumb"

    def test_no_duplicate_streaming_elements(self, html_content):
        """createStreamingMessage should check for existing element."""
        # The createStreamingMessage should handle the case where one already exists
        # This is handled by checking streamingElement before calling create
        assert "if (!streamingElement)" in html_content, \
            "Should check if streaming element already exists before creating"

    def test_streaming_content_field(self, html_content):
        """Should have streaming_content field for content updates."""
        # Check that the bubble element has a proper selector
        assert "streamingBubble" in html_content, \
            "Should have streamingBubble element for content"
        assert "querySelector('.message-bubble')" in html_content or \
               "querySelector(\".message-bubble\")" in html_content, \
               "Should query message-bubble for content update"
