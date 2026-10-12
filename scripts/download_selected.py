"""Select only Chinese corporate-report QA and referenced LaTeX from official ZIPs."""

import hashlib
import io
import json
import re
import time
import urllib.request
import zipfile
from collections import Counter, OrderedDict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/interim"
QUESTIONS_ZIP = WORK / "FinComplexQA-Pro.zip"
BASE_URL = "https://huggingface.co/datasets/Multilingual-Multimodal-NLP/FinanceComplexQA/resolve/main/"
SCENES = {
    "Corporate_Financial_Report": ("corporate_reports", 163),
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def download_questions():
    WORK.mkdir(parents=True, exist_ok=True)
    for directory in ('data/manifests', 'data/raw/financecomplexqa/questions/corporate_reports',
                      'data/raw/financecomplexqa/latex/corporate_reports'):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    if not QUESTIONS_ZIP.exists():
        urllib.request.urlretrieve(BASE_URL + "FinComplexQA-Pro.zip", QUESTIONS_ZIP)
    assert sha256(QUESTIONS_ZIP.read_bytes()) == "b8a7c93f7cb53adfe2a715b953af1e3a48fa557a9f47814c1854df960d5ca616"
    records = []
    files = []
    with zipfile.ZipFile(QUESTIONS_ZIP) as archive:
        for source_scene, (local_scene, expected) in SCENES.items():
            prefix = f"CN/scene_categories/{source_scene}/"
            members = [name for name in archive.namelist() if name.startswith(prefix) and name.endswith(".jsonl")]
            assert len(members) > 0, f"No files for {source_scene}"
            for member in members:
                raw = archive.read(member)
                target = ROOT / "data/raw/financecomplexqa/questions" / local_scene / Path(member).name
                target.write_bytes(raw)
                files.append({"source_path": member, "local_path": str(target.relative_to(ROOT)), "bytes": len(raw), "sha256": sha256(raw)})
                for number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    assert row.get("doc_type") == source_scene, (member, number, row.get("doc_type"))
                    records.append({
                        "scene": source_scene,
                        "local_scene": local_scene,
                        "finqa_id": row.get("finqa_id"),
                        "reference_document": row.get("Reference_documents"),
                        "source_question_file": member,
                    })
            count = sum(r["scene"] == source_scene for r in records)
            print(f"QUESTIONS {source_scene}: files={len(members)} rows={count} expected={expected}", flush=True)
            assert count == expected, f"Question count mismatch for {source_scene}: {count} vs {expected}"
    ids = [r["finqa_id"] for r in records]
    assert len(ids) == len(set(ids)), "Duplicate finqa_id in selected scenes"
    assert len(records) == 163
    manifest_dir = ROOT / "data/manifests"
    (manifest_dir / "questions.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    (manifest_dir / "question_files.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in files) + "\n", encoding="utf-8")
    return records


class RemoteZipReader(io.RawIOBase):
    def __init__(self, url, block_size=262144, max_blocks=32):
        self.source_url = url
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=45)
        self.url = head.url
        self.length = int(head.headers["Content-Length"])
        self.position = 0
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.cache = OrderedDict()
        self.bytes_fetched = 0
        print(f"REMOTE_ZIP size={self.length} url={self.url.split('?')[0]}", flush=True)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence == 0:
            self.position = offset
        elif whence == 1:
            self.position += offset
        elif whence == 2:
            self.position = self.length + offset
        else:
            raise ValueError(whence)
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def _block(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        start = index * self.block_size
        end = min(start + self.block_size, self.length) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        for attempt in range(5):
            try:
                response = urllib.request.urlopen(request, timeout=60)
                data = response.read()
                if response.status != 206 or len(data) != end - start + 1:
                    raise IOError(f"Unexpected range response: {response.status}, {len(data)}")
                self.bytes_fetched += len(data)
                self.cache[index] = data
                if len(self.cache) > self.max_blocks:
                    self.cache.popitem(last=False)
                return data
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)

    def read(self, amount=-1):
        if amount is None or amount < 0:
            amount = self.length - self.position
        amount = min(amount, self.length - self.position)
        parts = []
        left = amount
        while left:
            block_index = self.position // self.block_size
            block = self._block(block_index)
            offset = self.position % self.block_size
            piece = block[offset:offset + left]
            parts.append(piece)
            n = len(piece)
            self.position += n
            left -= n
        return b"".join(parts)


def select_latex(records):
    remote = RemoteZipReader(BASE_URL + "Reference_documents.zip")
    with zipfile.ZipFile(remote) as archive:
        names = archive.namelist()
        print(f"ZIP_MEMBERS={len(names)} SAMPLE={names[:5]}", flush=True)
        tex_names = [n for n in names if n.lower().endswith(".tex")]
        print(f"ZIP_TEX={len(tex_names)}", flush=True)
        (WORK / "reference_zip_tex_names.json").write_text(json.dumps(tex_names, ensure_ascii=False, indent=2), encoding="utf-8")

        # A reference path in QA can differ from the archive prefix. Match by full
        # normalized suffix first, then unique basename inside the scene directory.
        selected = []
        missing = []
        for scene, (local_scene, _) in SCENES.items():
            refs = sorted({r["reference_document"] for r in records if r["scene"] == scene})
            for ref in refs:
                if not isinstance(ref, str) or not ref.lower().endswith(".tex"):
                    missing.append({"scene": scene, "reference": ref, "reason": "not_tex"})
                    continue
                ref_norm = ref.replace("\\", "/")
                basename = ref_norm.rsplit("/", 1)[-1]
                matches = [n for n in tex_names if n.endswith("/" + basename) and ("/" + scene + "/") in ("/" + n)]
                if len(matches) != 1:
                    matches = [n for n in tex_names if n.endswith("/" + basename)]
                if len(matches) != 1:
                    missing.append({"scene": scene, "reference": ref, "matches": matches[:10]})
                    continue
                selected.append({"scene": scene, "local_scene": local_scene, "reference": ref, "zip_member": matches[0]})
            print(f"DOC_MAP {scene}: references={len(refs)} matched={sum(x['scene']==scene for x in selected)}", flush=True)

        (ROOT / "data/manifests/latex_mapping.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in selected) + "\n", encoding="utf-8")
        (ROOT / "data/manifests/latex_missing.json").write_text(json.dumps(missing, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"MISSING_REFERENCES={len(missing)}", flush=True)
        if missing:
            return
        seen = {}
        for index, entry in enumerate(selected, 1):
            key = (entry["local_scene"], entry["zip_member"])
            if key in seen:
                continue
            member = entry["zip_member"]
            raw = archive.read(member)
            target = ROOT / "data/raw/financecomplexqa/latex" / entry["local_scene"] / Path(member).name
            if target.exists() and sha256(target.read_bytes()) != sha256(raw):
                raise RuntimeError(f"Name collision: {target}")
            target.write_bytes(raw)
            seen[key] = {"scene": entry["scene"], "source_path": member, "local_path": str(target.relative_to(ROOT)), "bytes": len(raw), "sha256": sha256(raw)}
            if index % 25 == 0 or index == len(selected):
                print(f"TEX_PROGRESS {index}/{len(selected)} unique={len(seen)} fetched_bytes={remote.bytes_fetched}", flush=True)
        (ROOT / "data/manifests/latex_files.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in seen.values()) + "\n", encoding="utf-8")
        print(f"TEX_DONE unique={len(seen)} fetched_bytes={remote.bytes_fetched} zip_bytes={remote.length}", flush=True)


if __name__ == "__main__":
    records = download_questions()
    select_latex(records)

