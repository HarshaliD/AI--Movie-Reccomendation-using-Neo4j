"""Forwarder to canonical root streamlit_app.py.

Preserves backward-compatibility for workflows executing from inside movie-recommendation/.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from streamlit_app import main

if __name__ == "__main__":
    main()
