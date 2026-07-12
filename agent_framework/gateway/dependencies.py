"""Gateway dependencies - Dependency injection for FastAPI application.

This module provides dependency injection functions for the FastAPI
application, including session manager, memory, and other services.
"""
from typing import Optional

from ..core.session_manager import SessionManager


# Global state for the application
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> Optional[SessionManager]:
    """Get the current session manager instance.

    Returns:
        SessionManager instance if initialized, None otherwise.
    """
    return _session_manager


def set_session_manager(manager: SessionManager) -> None:
    """Set the session manager instance.

    Args:
        manager: SessionManager instance to set.
    """
    global _session_manager
    _session_manager = manager


def clear_session_manager() -> None:
    """Clear the session manager instance."""
    global _session_manager
    _session_manager = None
