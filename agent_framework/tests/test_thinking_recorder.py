"""Tests for ThinkingRecorder."""
import pytest
import time

from agent_framework.core.thinking_recorder import ThinkingRecorder


class TestThinkingRecorder:
    """Test suite for ThinkingRecorder class."""

    def test_init(self):
        """Test initialization."""
        recorder = ThinkingRecorder()
        assert recorder._current_step == 0
        assert recorder._steps == []
        assert recorder._start_time is None
        assert recorder._current_content == ""
        assert recorder._current_label == ""

    def test_start_thinking(self):
        """Test starting thinking."""
        recorder = ThinkingRecorder()
        thinking = recorder.start_thinking("test")

        assert thinking.step == 1
        assert thinking.label == "test"
        assert thinking.content == ""
        assert recorder._current_step == 1
        assert recorder._start_time is not None

    def test_start_thinking_without_label(self):
        """Test starting thinking without label."""
        recorder = ThinkingRecorder()
        thinking = recorder.start_thinking()

        assert thinking.step == 1
        assert thinking.label == ""
        assert thinking.content == ""

    def test_add_content(self):
        """Test adding content."""
        recorder = ThinkingRecorder()
        recorder.start_thinking()

        thinking = recorder.add_content("hello")
        assert thinking.content == "hello"
        assert thinking.step == 1

        thinking = recorder.add_content(" world")
        assert recorder._current_content == "hello world"

    def test_end_thinking(self):
        """Test ending thinking."""
        recorder = ThinkingRecorder()
        recorder.start_thinking()
        recorder.add_content("content")

        thinking = recorder.end_thinking()
        assert thinking.duration_ms is not None
        assert thinking.duration_ms >= 0
        assert recorder._start_time is None
        assert recorder._current_content == ""
        assert recorder._current_label == ""

    def test_multiple_steps(self):
        """Test multiple steps."""
        recorder = ThinkingRecorder()

        # Step 1
        recorder.start_thinking("step1")
        recorder.add_content("content1")
        recorder.end_thinking()

        # Step 2
        recorder.start_thinking("step2")
        recorder.add_content("content2")
        recorder.end_thinking()

        steps = recorder.get_all_steps()
        assert len(steps) == 2
        assert steps[0].step == 1
        assert steps[0].label == "step1"
        assert steps[0].content == "content1"
        assert steps[1].step == 2
        assert steps[1].label == "step2"
        assert steps[1].content == "content2"

    def test_get_all_steps_returns_copy(self):
        """Test that get_all_steps returns a copy."""
        recorder = ThinkingRecorder()
        recorder.start_thinking("step1")
        recorder.add_content("content1")
        recorder.end_thinking()

        steps = recorder.get_all_steps()
        steps.clear()  # Modify the returned list

        # Original should be unchanged
        assert len(recorder.get_all_steps()) == 1

    def test_get_summary(self):
        """Test getting summary."""
        recorder = ThinkingRecorder()

        recorder.start_thinking("step1")
        recorder.add_content("content1")
        recorder.end_thinking()

        summary = recorder.get_summary()
        assert "Step 1" in summary
        assert "content1" in summary

    def test_get_summary_empty(self):
        """Test getting summary when empty."""
        recorder = ThinkingRecorder()
        assert recorder.get_summary() == ""

    def test_get_summary_long_content(self):
        """Test getting summary with long content."""
        recorder = ThinkingRecorder()

        recorder.start_thinking("step1")
        long_content = "x" * 200
        recorder.add_content(long_content)
        recorder.end_thinking()

        summary = recorder.get_summary()
        assert "..." in summary
        assert len(summary) < 200  # Should be truncated

    def test_clear(self):
        """Test clearing."""
        recorder = ThinkingRecorder()

        recorder.start_thinking()
        recorder.add_content("content")
        recorder.end_thinking()

        recorder.clear()
        assert recorder._current_step == 0
        assert recorder._steps == []
        assert recorder._start_time is None
        assert recorder._current_content == ""
        assert recorder._current_label == ""

    def test_duration_calculation(self):
        """Test duration calculation."""
        recorder = ThinkingRecorder()

        recorder.start_thinking()
        time.sleep(0.1)  # Wait 100ms

        thinking = recorder.end_thinking()
        assert thinking.duration_ms is not None
        assert thinking.duration_ms >= 100

    def test_duration_none_before_start(self):
        """Test that duration is None before starting."""
        recorder = ThinkingRecorder()
        assert recorder._calculate_duration() is None

    def test_auto_save_previous_step(self):
        """Test that previous step is auto-saved when starting new step."""
        recorder = ThinkingRecorder()

        # Step 1
        recorder.start_thinking("step1")
        recorder.add_content("content1")
        # Don't call end_thinking

        # Step 2 - should auto-save step 1
        recorder.start_thinking("step2")

        steps = recorder.get_all_steps()
        assert len(steps) == 1
        assert steps[0].step == 1
        assert steps[0].label == "step1"
        assert steps[0].content == "content1"

    def test_end_thinking_without_content(self):
        """Test ending thinking without adding content."""
        recorder = ThinkingRecorder()
        recorder.start_thinking("step1")

        thinking = recorder.end_thinking()
        assert thinking.step == 1
        assert thinking.duration_ms is not None

        # Should not save empty step
        steps = recorder.get_all_steps()
        assert len(steps) == 0

    def test_add_content_without_start(self):
        """Test adding content without starting (should still work)."""
        recorder = ThinkingRecorder()

        # This should work but step will be 0
        thinking = recorder.add_content("orphan")
        assert thinking.step == 0
        assert thinking.content == "orphan"
