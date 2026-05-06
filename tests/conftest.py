import sys
from pathlib import Path

# Add src to PYTHONPATH so tests can import from it without installation
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
