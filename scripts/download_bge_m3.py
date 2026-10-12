"""Download BAAI's pinned official ONNX export, with upstream file verification."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'models/embedding/bge-m3-onnx'
REVISION = '5617a9f61b028005a4858fdac845db406aefb181'
API = f'https://huggingface.co/api/models/BAAI/bge-m3/revision/{REVISION}?blobs=true'
FILES = ['onnx/model.onnx', 'onnx/model.onnx_data', 'onnx/Constant_7_attr__value',
         'onnx/config.json', 'onnx/tokenizer.json', 'onnx/tokenizer_config.json',
         'onnx/special_tokens_map.json']

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def download_file(info):
    name = info['rfilename']
    out = DEST / Path(name).name
    def verified():
        if not out.is_file() or out.stat().st_size != info['size']: return False
        if info.get('lfs'): return digest(out) == info['lfs']['sha256']
        data = out.read_bytes()
        return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() == info['blobId']
    if verified(): return {'file': out.name, 'sha256': digest(out), 'bytes': out.stat().st_size}
    url = f'https://huggingface.co/BAAI/bge-m3/resolve/{REVISION}/{name}?download=true'
    if info['size'] > 128 * 1024 * 1024:
        size = 8 * 1024 * 1024
        count = (info['size']+size-1)//size
        def part(i):
            start=i*size;end=min(info['size'],start+size)-1
            p=DEST / f'{out.name}.part{i:04d}'
            if p.exists() and p.stat().st_size==end-start+1:return p
            for attempt in range(4):
                try:
                    req=urllib.request.Request(url+f'&part={i}',headers={'Range':f'bytes={start}-{end}'})
                    with urllib.request.urlopen(req,timeout=60) as response:
                        if response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
                            raise ValueError('Incorrect requested download range')
                        with p.open('wb') as f:
                            while block:=response.read(1024*1024):f.write(block)
                    if p.stat().st_size!=end-start+1:raise ValueError('Part size mismatch')
                    return p
                except Exception:
                    p.unlink(missing_ok=True)
                    if attempt==3:raise
                    time.sleep(2**attempt)
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get('BGE_DOWNLOAD_WORKERS','16'))) as pool:
            futures=[pool.submit(part,i) for i in range(count)]
            for n,future in enumerate(concurrent.futures.as_completed(futures),1):
                future.result()
                if n%16==0 or n==count:print(f'{name}: verified range downloads {n}/{count}',flush=True)
        partial=out.with_suffix(out.suffix+'.assembled')
        with partial.open('wb') as f:
            for i in range(count):
                with (DEST/f'{out.name}.part{i:04d}').open('rb') as source:
                    while block:=source.read(4*1024*1024):f.write(block)
        partial.replace(out)
        if not verified():out.unlink();raise ValueError('Upstream checksum mismatch after range download')
        for i in range(count):(DEST/f'{out.name}.part{i:04d}').unlink()
        out.with_suffix(out.suffix+'.partial').unlink(missing_ok=True)
        return {'file':out.name,'sha256':digest(out),'bytes':out.stat().st_size}
    partial = out.with_suffix(out.suffix + '.partial')
    for attempt in range(4):
        try:
            start = partial.stat().st_size if partial.exists() else 0
            headers = {'Range': f'bytes={start}-'} if start else {}
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                append = bool(start and response.status == 206)
                if append and not response.headers.get('Content-Range', '').startswith(f'bytes {start}-'):
                    raise ValueError('Incorrect resume range')
                done = start if append else 0
                last_report = done
                with partial.open('ab' if append else 'wb') as f:
                    while block := response.read(4 * 1024 * 1024):
                        f.write(block); done += len(block)
                        if done - last_report >= 128 * 1024 * 1024:
                            print(f'{name}: {done}/{info["size"]} bytes ({done/info["size"]:.1%})', flush=True)
                            last_report = done
            if partial.stat().st_size != info['size']: raise ValueError('Download size mismatch')
            partial.replace(out)
            if not verified():
                out.unlink()
                raise ValueError('Upstream checksum mismatch')
            print(f'Verified {name}: {out.stat().st_size} bytes', flush=True)
            return {'file': out.name, 'sha256': digest(out), 'bytes': out.stat().st_size}
        except Exception as exc:
            print(f'{name}: retry {attempt+1}: {type(exc).__name__}: {exc}', flush=True)
            if attempt == 3: raise
            time.sleep(2 ** attempt)

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(API, timeout=30) as response: model = json.load(response)
    if model['sha'] != REVISION: raise ValueError('Upstream revision mismatch')
    infos = {x['rfilename']: x for x in model['siblings']}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(download_file, [infos[x] for x in FILES]))
    (DEST / 'download_manifest.json').write_text(json.dumps(dict(model='BAAI/bge-m3',
        revision=REVISION, origin='official BAAI ONNX export', files=results), indent=2)+'\n', encoding='utf-8')
    print('BGE-M3 download and upstream verification complete', flush=True)

if __name__ == '__main__': main()
