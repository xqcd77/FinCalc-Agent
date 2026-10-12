"""LoRA entry point with an explicit, portable default profile."""
import argparse
import runpy
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', default=str(ROOT / 'configs/lora.json'))
    parser.add_argument('--output-dir')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--resume-from-checkpoint')
    parser.parse_known_args()  # --help works without installing GPU libraries.
    if not any(x == '--profile' or x.startswith('--profile=') for x in sys.argv):
        sys.argv.extend(['--profile', str(ROOT / 'configs/lora.json')])
    try:
        runpy.run_path(str(ROOT / 'scripts/train_lora_qwen35_9b_5250.py'), run_name='__main__')
    except ModuleNotFoundError as exc:
        raise SystemExit('Missing training dependency: ' + str(exc.name) +
                         '. Install CUDA PyTorch and requirements/training.txt.') from None
