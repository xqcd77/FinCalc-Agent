"""Portable entry point for the single-Agent RAG workflow."""
import runpy
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/rag'))
if __name__ == '__main__':
    runpy.run_path(str(ROOT / 'src/rag/agent_v4.py'), run_name='__main__')
