import os
import sys
from pathlib import Path

os.environ.setdefault("HPV_TRACE_STDOUT", "0")
os.environ.setdefault("HPV_PLANNER", "scripted")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
