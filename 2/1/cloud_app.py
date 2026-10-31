"""Streamlit Community Cloud入口；本机仍使用app.py。"""
import os
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parent
os.environ["GUANLAN_CLOUD"] = "1"
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
runpy.run_path(str(root / "app.py"), run_name="__main__")
