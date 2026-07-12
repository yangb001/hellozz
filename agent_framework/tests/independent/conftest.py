"""Pytest configuration for independent tests.

This conftest is specific to independent verification tests and does not
depend on the main test fixtures.
"""
import pytest
import sys
import os

# 添加 agent-framework 目录到 Python 路径（用于 infrastructure.storage.vector_store 导入）
agent_framework_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if agent_framework_dir not in sys.path:
    sys.path.insert(0, agent_framework_dir)

# 同时添加父目录（用于可能的 agent_framework 导入）
parent_dir = os.path.dirname(agent_framework_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
