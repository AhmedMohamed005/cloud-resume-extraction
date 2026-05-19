"""Default test env: avoid loading NER weights unless a test overrides."""
import os

os.environ.setdefault("USE_MOCK_INFERENCE", "1")
