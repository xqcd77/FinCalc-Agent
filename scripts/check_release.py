"""Check syntax, local document links, release sizes and credential patterns."""
import ast
import json
from pathlib import Path
import re
import subprocess
ROOT = Path(__file__).resolve().parents[1]
SECRET_RULES = {
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'service_api_key': re.compile(r'\bsk-[A-Za-z0-9]{20,}\b'),
    'github_token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b'),
    'credential_url': re.compile(r'https?://[^/\s:@]+:[^/\s@]+@'),
    'personal_windows_path': re.compile(r'[A-Za-z]:[\\/]Users[\\/](?!Public\b)[^\s"\']+'),
}
def files():
    if (ROOT/'.git').is_dir():
        result = subprocess.run(['git','ls-files','--cached','--others','--exclude-standard'], cwd=ROOT,
                                capture_output=True,text=True,check=True)
        return sorted({ROOT/line for line in result.stdout.splitlines() if line})
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and not any(
        part in {'.git','.venv','__pycache__','.pytest_cache'} for part in p.parts)
        and p.name != 'release_validation.local.json')
def main():
    issues = []
    paths = files()
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        if path.stat().st_size > 5*1024*1024: issues.append({'file':rel,'rule':'large_release_file'})
        try: text = path.read_text(encoding='utf-8')
        except (UnicodeDecodeError,OSError):
            issues.append({'file':rel,'rule':'unexpected_binary'}); continue
        if path.suffix == '.py':
            try: ast.parse(text, filename=rel)
            except SyntaxError: issues.append({'file':rel,'rule':'syntax'})
        for rule, pattern in SECRET_RULES.items():
            if pattern.search(text): issues.append({'file':rel,'rule':rule})
        if path.suffix == '.md':
            for link in re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)', text):
                if '://' in link or link.startswith('#'): continue
                target = link.split('#',1)[0]
                if target and not (path.parent/target).exists(): issues.append({'file':rel,'rule':'broken_link','target':target})
    report = {'files':len(paths),'python_files':sum(p.suffix=='.py' for p in paths),
              'bytes':sum(p.stat().st_size for p in paths),'issues':issues,'passed':not issues}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 1 if issues else 0
if __name__ == '__main__':
    raise SystemExit(main())
