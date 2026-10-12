"""Install only additive pinned kernels; never upgrade torch/transformers/peft."""
import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import urllib.request
from pathlib import Path
import torch

def fetch(url):
    req=urllib.request.Request(url,headers={'User-Agent':'FinCalc-kernel-preparation'})
    with urllib.request.urlopen(req,timeout=60) as r:return r.read()

def package_versions():
    return {p:importlib.metadata.version(p) for p in ['torch','transformers','peft','triton','bitsandbytes']}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir',required=True)
    ap.add_argument('--install',action='store_true')
    args=ap.parse_args()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    before=package_versions()
    fla=json.loads(fetch('https://pypi.org/pypi/flash-linear-attention/0.5.2/json'))
    core=json.loads(fetch('https://pypi.org/pypi/fla-core/0.5.2/json'))
    # Pin a tested-release family. --no-deps prevents hidden framework changes.
    pinned={'flash-linear-attention':'0.5.2','fla-core':'0.5.2'}
    torch_tag='torch'+'.'.join(torch.__version__.split('+')[0].split('.')[:2])
    py_tag=f'cp{sys.version_info.major}{sys.version_info.minor}'
    abi='TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE'
    suffix=f'cu12{torch_tag}cxx11abi{abi}-{py_tag}-{py_tag}-linux_x86_64.whl'
    releases=json.loads(fetch('https://api.github.com/repos/Dao-AILab/causal-conv1d/releases?per_page=12'))
    assets=[(r,a) for r in releases for a in r.get('assets',[]) if a['name'].endswith(suffix)]
    if not assets:raise RuntimeError('No official matching causal-conv1d wheel: '+suffix)
    release,asset=assets[0]
    record=dict(framework_versions_before=before,fla_dependencies=fla['info'].get('requires_dist'),
                core_dependencies=core['info'].get('requires_dist'),pinned=pinned,
                causal_release=release['tag_name'],causal_asset=asset['name'],causal_url=asset['browser_download_url'],
                install_requested=args.install)
    (out/'kernel_install_plan.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)
    if args.install:
        subprocess.run([sys.executable,'-m','pip','install','--no-deps','flash-linear-attention==0.5.2','fla-core==0.5.2'],check=True)
        wheel=out/asset['name']
        if not wheel.exists():wheel.write_bytes(fetch(asset['browser_download_url']))
        record['wheel_sha256']=hashlib.sha256(wheel.read_bytes()).hexdigest()
        expected=asset.get('digest')
        if expected and expected.startswith('sha256:') and expected[7:]!=record['wheel_sha256']:
            raise RuntimeError('Official wheel digest mismatch')
        subprocess.run([sys.executable,'-m','pip','install','--no-deps',str(wheel)],check=True)
        record['framework_versions_after']=package_versions()
        if record['framework_versions_after']!=before:raise RuntimeError('Framework versions changed unexpectedly')
        record['installed_versions']={p:importlib.metadata.version(p) for p in ['flash-linear-attention','fla-core','causal-conv1d']}
        (out/'kernel_install_result.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record,indent=2),flush=True)

if __name__=='__main__':main()
