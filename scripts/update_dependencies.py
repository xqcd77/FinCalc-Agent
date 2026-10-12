"""Regenerate the relative-path import map without importing GPU modules."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    files = sorted(list(ROOT.glob('src/**/*.py')) + list(ROOT.glob('scripts/*.py')) + list(ROOT.glob('tests/*.py')) + list(ROOT.glob('examples/*.py')))
    modules = {p.stem: p.relative_to(ROOT).as_posix() for p in files}
    result = []
    for path in files:
        names = set()
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name.split('.')[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split('.')[0])
        result.append({'file': path.relative_to(ROOT).as_posix(),
                       'local_imports': sorted(modules[n] for n in names if n in modules),
                       'external_or_stdlib': sorted(n for n in names if n not in modules)})
    output = ROOT / 'docs/dependencies.json'
    output.write_text(json.dumps({'modules':result}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'modules':len(result), 'output':'docs/dependencies.json'}))

if __name__ == '__main__':
    main()
