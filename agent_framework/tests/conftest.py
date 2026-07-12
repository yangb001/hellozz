"""Pytest configuration and fixtures for agent-framework tests."""
import pytest
import os
import tempfile


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
async def storage(temp_db):
    """Create a SQLiteSessionStorage instance with temp DB."""
    from agent_framework.infrastructure.storage.sqlite_session_storage import SQLiteSessionStorage
    st = SQLiteSessionStorage(db_path=temp_db)
    # Trigger database file creation
    await st.save(SessionContext(session_id="__init__"))
    await st.delete("__init__")
    return st


# Import these for type hints in fixtures
from agent_framework.interfaces.session import SessionContext
