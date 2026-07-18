"""Thinking Recorder - Record LLM thinking process."""
import time
from typing import List, Optional

from agent_framework.interfaces.events import ThinkingData


class ThinkingRecorder:
    """Record LLM thinking process.

    Used to capture and record LLM reasoning process, supporting:
    - Multi-step thinking
    - Duration statistics
    - Content accumulation
    """

    def __init__(self):
        """Initialize thinking recorder."""
        self._steps: List[ThinkingData] = []
        self._current_step = 0
        self._start_time: Optional[float] = None
        self._current_content = ""
        self._current_label = ""

    def start_thinking(self, label: str = "") -> ThinkingData:
        """Start a new thinking step.

        Args:
            label: Step label, e.g., "analyze problem", "select tool".

        Returns:
            ThinkingData object.
        """
        # Save previous step if exists
        if self._current_step > 0 and self._current_content:
            self._steps.append(ThinkingData(
                step=self._current_step,
                label=self._current_label,
                content=self._current_content,
                duration_ms=self._calculate_duration()
            ))

        self._current_step += 1
        self._start_time = time.time()
        self._current_content = ""
        self._current_label = label

        return ThinkingData(
            step=self._current_step,
            label=label,
            content=""
        )

    def add_content(self, content: str) -> ThinkingData:
        """Add thinking content.

        Args:
            content: Thinking content fragment.

        Returns:
            ThinkingData object.
        """
        self._current_content += content

        return ThinkingData(
            step=self._current_step,
            label=self._current_label,
            content=content,
            duration_ms=self._calculate_duration()
        )

    def end_thinking(self) -> ThinkingData:
        """End current thinking step.

        Returns:
            ThinkingData object.
        """
        duration_ms = self._calculate_duration()

        # Save current step
        if self._current_content:
            self._steps.append(ThinkingData(
                step=self._current_step,
                label=self._current_label,
                content=self._current_content,
                duration_ms=duration_ms
            ))

        self._start_time = None
        self._current_content = ""
        self._current_label = ""

        return ThinkingData(
            step=self._current_step,
            content="",
            duration_ms=duration_ms
        )

    def get_all_steps(self) -> List[ThinkingData]:
        """Get all thinking steps.

        Returns:
            List of ThinkingData objects.
        """
        return self._steps.copy()

    def get_summary(self) -> str:
        """Get thinking summary.

        Returns:
            Summary text.
        """
        if not self._steps:
            return ""

        summary_parts = []
        for step in self._steps:
            if step.content:
                content_preview = step.content[:100]
                if len(step.content) > 100:
                    content_preview += "..."
                summary_parts.append(
                    f"Step {step.step}: {content_preview}"
                )

        return "\n".join(summary_parts)

    def clear(self):
        """Clear all records."""
        self._steps.clear()
        self._current_step = 0
        self._start_time = None
        self._current_content = ""
        self._current_label = ""

    def _calculate_duration(self) -> Optional[int]:
        """Calculate duration.

        Returns:
            Duration in milliseconds, or None if not started.
        """
        if self._start_time is None:
            return None
        return int((time.time() - self._start_time) * 1000)
