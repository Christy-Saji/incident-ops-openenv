"""pytest configuration — makes the project root importable from tests/."""
import sys
import os

# Ensure the project root is on sys.path so tests can import env, graders, tasks, training
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
