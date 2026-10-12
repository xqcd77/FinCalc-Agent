"""Offline evaluation; reference answers are never passed to generation."""
import runpy
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src/eval'), str(ROOT / 'src/rag')]
if __name__ == '__main__':
    runpy.run_path(str(ROOT / 'src/eval/evaluate_v48.py'), run_name='__main__')
