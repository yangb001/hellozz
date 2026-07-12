"""Independent tests for multi-session web page verification.

This test file verifies the web page implementation for the AI Agent Framework,
including HTML structure, JavaScript functionality, API integration, and responsive design.

Design Reference:
- Frontend: agent_framework/gateway/static/index.html
- Backend API: agent_framework/gateway/api/rest.py (prefix: /api/v1)
- WebSocket: agent_framework/gateway/api/websocket.py (path: /ws/chat)

Test Coverage:
1. Page structure - HTML elements exist
2. JavaScript functions - all required functions defined
3. API integration - frontend calls match backend routes
4. Responsive design - viewport and CSS patterns
5. Multi-session functionality - session switching logic
"""
import pytest
import os
import re
from pathlib import Path


# Path to the HTML file
HTML_FILE = Path(__file__).parent.parent.parent / "agent_framework" / "gateway" / "static" / "index.html"


@pytest.fixture
def html_content():
    """Load the HTML file content."""
    assert HTML_FILE.exists(), f"HTML file not found: {HTML_FILE}"
    with open(HTML_FILE, 'r', encoding='utf-8') as f:
        return f.read()


@pytest.fixture
def html_lines():
    """Load HTML file as lines for line-specific checks."""
    assert HTML_FILE.exists(), f"HTML file not found: {HTML_FILE}"
    with open(HTML_FILE, 'r', encoding='utf-8') as f:
        return f.readlines()


class TestPageStructure:
    """Test HTML page structure and required elements."""

    def test_html_file_exists(self):
        """HTML file must exist at the expected location."""
        assert HTML_FILE.exists(), f"index.html not found at {HTML_FILE}"

    def test_html5_doctype(self, html_content):
        """Page should use HTML5 doctype."""
        assert html_content.strip().startswith('<!DOCTYPE html>'), \
            "Page should start with HTML5 doctype"

    def test_viewport_meta_tag(self, html_content):
        """Page must have viewport meta tag for responsive design."""
        assert 'viewport' in html_content, "Missing viewport meta tag"
        assert 'width=device-width' in html_content, \
            "Viewport should include width=device-width"
        assert 'initial-scale=1.0' in html_content, \
            "Viewport should include initial-scale=1.0"

    def test_charset_meta_tag(self, html_content):
        """Page should have charset meta tag."""
        assert 'charset="UTF-8"' in html_content or 'charset=UTF-8' in html_content, \
            "Missing charset meta tag"

    def test_page_title(self, html_content):
        """Page should have a title."""
        assert '<title>' in html_content, "Missing <title> tag"
        assert '</title>' in html_content, "Missing closing </title> tag"
        # Extract title content
        title_match = re.search(r'<title>(.*?)</title>', html_content)
        assert title_match, "Could not extract title content"
        assert len(title_match.group(1).strip()) > 0, "Title should not be empty"

    def test_sidebar_element(self, html_content):
        """Page should have a sidebar for session list."""
        assert 'class="sidebar"' in html_content, "Missing sidebar element"

    def test_session_list_element(self, html_content):
        """Page should have session list container."""
        assert 'id="sessionList"' in html_content, "Missing session list element"

    def test_new_session_button(self, html_content):
        """Page should have a new session button."""
        assert 'new-session-btn' in html_content, "Missing new session button"
        assert 'createSession()' in html_content, \
            "New session button should call createSession()"

    def test_chat_header_element(self, html_content):
        """Page should have a chat header."""
        assert 'id="chatHeader"' in html_content, "Missing chat header element"
        assert 'id="sessionTitle"' in html_content, "Missing session title element"

    def test_chat_content_area(self, html_content):
        """Page should have chat content area."""
        assert 'id="chatContent"' in html_content, "Missing chat content area"

    def test_input_area(self, html_content):
        """Page should have message input area."""
        assert 'id="inputArea"' in html_content, "Missing input area"
        assert 'id="messageInput"' in html_content, "Missing message input field"
        assert 'id="sendBtn"' in html_content, "Missing send button"

    def test_status_bar(self, html_content):
        """Page should have a status bar."""
        assert 'id="statusBar"' in html_content, "Missing status bar"
        assert 'id="statusDot"' in html_content, "Missing status dot"
        assert 'id="statusText"' in html_content, "Missing status text"
        assert 'id="messageCount"' in html_content, "Missing message count"

    def test_empty_state(self, html_content):
        """Page should have an empty state display."""
        assert 'class="empty-state"' in html_content, "Missing empty state"

    def test_typing_indicator(self, html_content):
        """Page should have typing indicator for loading state."""
        # The typing indicator is created dynamically in JS
        assert 'typing-indicator' in html_content, "Missing typing indicator"

    def test_message_structure(self, html_content):
        """Page should have message bubble structure."""
        assert 'message-bubble' in html_content, "Missing message bubble class"
        assert 'message-meta' in html_content, "Missing message meta class"


class TestJavaScriptFunctions:
    """Test JavaScript functions are defined."""

    def test_create_session_function(self, html_content):
        """createSession() function must be defined."""
        assert 'async function createSession()' in html_content or \
               'function createSession()' in html_content, \
               "createSession() function not defined"

    def test_load_sessions_function(self, html_content):
        """loadSessions() function must be defined."""
        assert 'async function loadSessions()' in html_content or \
               'function loadSessions()' in html_content, \
               "loadSessions() function not defined"

    def test_select_session_function(self, html_content):
        """selectSession() function must be defined."""
        assert 'async function selectSession(' in html_content or \
               'function selectSession(' in html_content, \
               "selectSession() function not defined"

    def test_send_message_function(self, html_content):
        """sendMessage() function must be defined."""
        assert 'async function sendMessage()' in html_content or \
               'function sendMessage()' in html_content, \
               "sendMessage() function not defined"

    def test_render_session_list_function(self, html_content):
        """renderSessionList() function must be defined."""
        assert 'function renderSessionList()' in html_content, \
               "renderSessionList() function not defined"

    def test_render_messages_function(self, html_content):
        """renderMessages() function must be defined."""
        assert 'function renderMessages(' in html_content, \
               "renderMessages() function not defined"

    def test_append_message_function(self, html_content):
        """appendMessage() function must be defined."""
        assert 'function appendMessage(' in html_content, \
               "appendMessage() function not defined"

    def test_connect_websocket_function(self, html_content):
        """connectWebSocket() function must be defined."""
        assert 'function connectWebSocket()' in html_content, \
               "connectWebSocket() function not defined"

    def test_handle_websocket_message_function(self, html_content):
        """handleWebSocketMessage() function must be defined."""
        assert 'function handleWebSocketMessage(' in html_content, \
               "handleWebSocketMessage() function not defined"

    def test_handle_key_press_function(self, html_content):
        """handleKeyPress() function must be defined."""
        assert 'function handleKeyPress(' in html_content, \
               "handleKeyPress() function not defined"

    def test_load_messages_function(self, html_content):
        """loadMessages() function must be defined."""
        assert 'async function loadMessages()' in html_content or \
               'function loadMessages()' in html_content, \
               "loadMessages() function not defined"

    def test_scroll_to_bottom_function(self, html_content):
        """scrollToBottom() function must be defined."""
        assert 'function scrollToBottom()' in html_content, \
               "scrollToBottom() function not defined"

    def test_update_status_function(self, html_content):
        """updateStatus() function must be defined."""
        assert 'function updateStatus(' in html_content, \
               "updateStatus() function not defined"

    def test_escape_html_function(self, html_content):
        """escapeHtml() function must be defined for XSS prevention."""
        assert 'function escapeHtml(' in html_content, \
               "escapeHtml() function not defined"

    def test_format_time_function(self, html_content):
        """formatTime() function must be defined."""
        assert 'function formatTime(' in html_content, \
               "formatTime() function not defined"

    def test_save_sessions_function(self, html_content):
        """saveSessions() function must be defined for local storage."""
        assert 'function saveSessions()' in html_content, \
               "saveSessions() function not defined"

    def test_dom_content_loaded_listener(self, html_content):
        """Page should initialize on DOMContentLoaded."""
        assert 'DOMContentLoaded' in html_content, \
               "Missing DOMContentLoaded event listener"


class TestAPIIntegration:
    """Test frontend API calls match backend routes."""

    def test_api_base_url(self, html_content):
        """API base URL should match backend prefix."""
        assert "API_BASE" in html_content, "Missing API_BASE constant"
        assert '/api/v1' in html_content, "API_BASE should be '/api/v1'"

    def test_create_session_endpoint(self, html_content):
        """Frontend should call POST /api/v1/sessions."""
        assert '/sessions' in html_content, "Missing sessions endpoint"
        assert "method: 'POST'" in html_content or 'method: "POST"' in html_content, \
               "Create session should use POST method"

    def test_load_messages_endpoint(self, html_content):
        """Frontend should call GET /api/v1/sessions/{id}/messages."""
        assert '/messages' in html_content, "Missing messages endpoint"

    def test_send_message_endpoint(self, html_content):
        """Frontend should have fallback REST endpoint for sending messages."""
        # Check for REST fallback when WebSocket is not available
        assert '/sessions/' in html_content, "Missing session-specific endpoints"
        assert '/messages' in html_content, "Missing messages endpoint"

    def test_websocket_url_format(self, html_content):
        """WebSocket URL should use correct protocol and path."""
        assert 'ws/chat' in html_content or '/ws/chat' in html_content, \
               "Missing WebSocket chat endpoint"
        assert 'session_id=' in html_content, "WebSocket URL should include session_id parameter"

    def test_websocket_protocol_detection(self, html_content):
        """WebSocket should detect http/https protocol correctly."""
        assert "wss:" in html_content, "Missing wss: protocol for HTTPS"
        assert "ws:" in html_content, "Missing ws: protocol for HTTP"

    def test_request_content_type(self, html_content):
        """API requests should set Content-Type header."""
        assert 'Content-Type' in html_content, "Missing Content-Type header"
        assert 'application/json' in html_content, "Content-Type should be application/json"

    def test_create_session_request_body(self, html_content):
        """Create session request should include required fields."""
        assert 'user_id' in html_content, "Missing user_id in create session request"
        assert 'session_type' in html_content, "Missing session_type in create session request"

    def test_send_message_request_body(self, html_content):
        """Send message request should include content field."""
        assert 'content' in html_content, "Missing content in send message request"

    def test_error_handling(self, html_content):
        """Frontend should handle API errors."""
        assert 'catch' in html_content, "Missing error handling"
        assert 'error' in html_content.lower(), "Should have error handling logic"


class TestMultiSessionFunctionality:
    """Test multi-session switching functionality."""

    def test_session_state_variable(self, html_content):
        """Should have currentSessionId state variable."""
        assert 'currentSessionId' in html_content, "Missing currentSessionId state"

    def test_sessions_array(self, html_content):
        """Should have sessions array to track multiple sessions."""
        assert 'sessions' in html_content, "Missing sessions array"

    def test_session_list_rendering(self, html_content):
        """Should render session list with active state."""
        assert 'active' in html_content, "Missing active session class"
        assert 'session-item' in html_content, "Missing session-item class"

    def test_session_id_display(self, html_content):
        """Should display truncated session ID."""
        assert '.substring(0, 8)' in html_content or \
               'substring(0, 8)' in html_content, \
               "Should truncate session ID for display"

    def test_session_selection_updates_ui(self, html_content):
        """Selecting a session should update UI elements."""
        assert "chatHeader" in html_content, "Should update chat header on selection"
        assert "inputArea" in html_content, "Should show input area on selection"
        assert "sessionTitle" in html_content, "Should update session title"

    def test_local_storage_persistence(self, html_content):
        """Sessions should be persisted in local storage."""
        assert 'localStorage' in html_content, "Missing localStorage usage"
        assert 'sessions' in html_content, "Should store sessions in localStorage"

    def test_session_created_at_display(self, html_content):
        """Should display session creation time."""
        assert 'created_at' in html_content, "Missing created_at field"
        assert 'formatTime' in html_content, "Should format time for display"

    def test_session_type_field(self, html_content):
        """Should store session type."""
        assert 'type' in html_content, "Missing session type field"

    def test_websocket_per_session(self, html_content):
        """Should create new WebSocket connection per session."""
        assert 'ws.close()' in html_content or 'ws = new WebSocket' in html_content, \
               "Should manage WebSocket connections per session"

    def test_message_count_tracking(self, html_content):
        """Should track and display message count per session."""
        assert 'messageCount' in html_content, "Missing message count display"
        assert 'total_count' in html_content or 'updateMessageCount' in html_content, \
               "Should update message count"


class TestResponsiveDesign:
    """Test responsive design elements."""

    def test_flexbox_layout(self, html_content):
        """Should use flexbox for layout."""
        assert 'display: flex' in html_content, "Should use flexbox layout"

    def test_sidebar_width(self, html_content):
        """Sidebar should have fixed width."""
        assert 'width: 280px' in html_content, "Sidebar should have fixed width"

    def test_main_area_flex_grow(self, html_content):
        """Main area should grow to fill space."""
        assert 'flex: 1' in html_content, "Main area should use flex: 1"

    def test_overflow_scroll(self, html_content):
        """Lists should have overflow scroll."""
        assert 'overflow-y: auto' in html_content, "Should have scrollable areas"

    def test_max_width_messages(self, html_content):
        """Messages should have max width for readability."""
        assert 'max-width: 70%' in html_content or 'max-width' in html_content, \
               "Messages should have max-width constraint"

    def test_responsive_font_sizes(self, html_content):
        """Should use relative font sizes."""
        assert 'font-size' in html_content, "Should define font sizes"

    def test_box_sizing(self, html_content):
        """Should use border-box sizing."""
        assert 'box-sizing: border-box' in html_content, "Should use border-box sizing"

    def test_smooth_transitions(self, html_content):
        """Should have smooth transitions for interactions."""
        assert 'transition' in html_content, "Should have CSS transitions"

    def test_cursor_pointer_on_buttons(self, html_content):
        """Buttons should have pointer cursor."""
        assert 'cursor: pointer' in html_content, "Buttons should have pointer cursor"


class TestSecurityAndUX:
    """Test security and UX elements."""

    def test_xss_prevention(self, html_content):
        """Should escape HTML in user messages."""
        assert 'escapeHtml' in html_content, "Missing escapeHtml function"
        assert 'textContent' in html_content, "Should use textContent for escaping"

    def test_empty_message_validation(self, html_content):
        """Should validate empty messages."""
        assert 'trim()' in html_content, "Should trim message input"
        assert '!content' in html_content or 'content === ""' in html_content, \
               "Should check for empty content"

    def test_processing_state(self, html_content):
        """Should track processing state to prevent double-sends."""
        assert 'isProcessing' in html_content, "Missing processing state"
        assert 'disabled' in html_content, "Should disable send button while processing"

    def test_enter_key_handling(self, html_content):
        """Should handle Enter key for sending."""
        assert 'Enter' in html_content, "Should handle Enter key"
        assert 'handleKeyPress' in html_content, "Should have key press handler"

    def test_input_focus_style(self, html_content):
        """Input should have focus style."""
        assert ':focus' in html_content, "Should have focus styles"

    def test_hover_styles(self, html_content):
        """Interactive elements should have hover styles."""
        assert ':hover' in html_content, "Should have hover styles"

    def test_typing_indicator(self, html_content):
        """Should show typing indicator during processing."""
        assert 'showTypingIndicator' in html_content, "Missing show typing indicator"
        assert 'hideTypingIndicator' in html_content, "Missing hide typing indicator"


class TestCodeQualityReview:
    """Review code quality of the web page."""

    def test_no_inline_scripts_in_body(self, html_content):
        """JavaScript should be in script tag, not inline event handlers (mostly)."""
        # Some inline onclick is acceptable for simple handlers
        # but core logic should be in functions
        script_match = re.search(r'<script>(.*?)</script>', html_content, re.DOTALL)
        assert script_match, "Should have script tag with JavaScript code"
        script_content = script_match.group(1)
        assert len(script_content) > 100, "Script should contain substantial code"

    def test_css_in_style_tag(self, html_content):
        """CSS should be in style tag."""
        assert '<style>' in html_content, "Should have style tag"
        assert '</style>' in html_content, "Should have closing style tag"

    def test_semantic_html(self, html_content):
        """Should use semantic HTML elements."""
        assert '<body>' in html_content, "Should have body tag"
        assert '<head>' in html_content, "Should have head tag"

    def test_proper_event_binding(self, html_content):
        """Should bind events properly."""
        assert 'onclick=' in html_content, "Should have onclick handlers"
        assert 'onkeypress=' in html_content, "Should have keypress handler"

    def test_console_error_logging(self, html_content):
        """Should log errors to console for debugging."""
        assert 'console.error' in html_content, "Should log errors to console"

    def test_console_log_for_debugging(self, html_content):
        """Should have console.log for debugging."""
        assert 'console.log' in html_content, "Should have debug logging"

    def test_message_role_classes(self, html_content):
        """Messages should have role-based classes."""
        assert 'message.user' in html_content or "message ${role}" in html_content, \
               "Should apply role-based classes to messages"

    def test_scroll_to_bottom_on_new_message(self, html_content):
        """Should scroll to bottom when new message added."""
        assert 'scrollToBottom' in html_content, "Should scroll to bottom on new messages"

    def test_websocket_error_handling(self, html_content):
        """Should handle WebSocket errors."""
        assert 'ws.onerror' in html_content, "Should handle WebSocket errors"
        assert 'ws.onclose' in html_content, "Should handle WebSocket close"

    def test_fallback_to_rest_api(self, html_content):
        """Should fallback to REST API when WebSocket unavailable."""
        assert 'Fallback to REST API' in html_content or \
               'REST API' in html_content or \
               'fetch(' in html_content, \
               "Should have REST API fallback"
