"""Content fingerprints for the fixed weights actually loaded by a server."""
import hashlib
from pathlib import Path


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def model_artifacts(path):
    root = Path(path)
    files = [p for p in root.rglob('*') if p.is_file() and '.git' not in p.parts
             and p.suffix in {'.safetensors', '.bin', '.json', '.jinja', '.py', '.model', '.txt'}]
    if not any(p.suffix in {'.safetensors', '.bin'} for p in files):
        raise ValueError('Model/adapter fingerprint requires weight files')
    return {str(p.relative_to(root)).replace('\\', '/'): hash_file(p) for p in sorted(files)}
