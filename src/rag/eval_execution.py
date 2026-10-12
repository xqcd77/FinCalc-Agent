"""Bounded CPU overlap and fingerprint-checked, single-writer result journal."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
from exact_cpu_cache import signature


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def path_hashes(root, suffixes=None):
    root = Path(root)
    if not root.exists(): raise FileNotFoundError(root)
    files = [root] if root.is_file() else sorted(p for p in root.rglob('*') if p.is_file()
        and '__pycache__' not in p.parts and (suffixes is None or p.suffix in suffixes))
    return {str(p.relative_to(root) if root.is_dir() else p.name).replace('\\','/'): file_hash(p) for p in files}


def bounded_results(rows, run, workers=1):
    """At most two live question states; output completion order for journaling."""
    if workers not in (1, 2): raise ValueError('question workers must be 1 or 2')
    if workers == 1:
        for row in rows: yield run(row)
        return
    iterator = iter(rows)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='rag-question') as pool:
        pending = {}
        for row in iterator:
            pending[pool.submit(run, row)] = row
            if len(pending) == workers: break
        while pending:
            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                pending.pop(future)
                yield future.result()
                row = next(iterator, None)
                if row is not None: pending[pool.submit(run, row)] = row


class ResultJournal:
    def __init__(self, output, ids, identity, resume=False, retry_api_errors=False):
        self.output = Path(output)
        self.partial = self.output.with_suffix('.jsonl.partial')
        self.manifest = self.output.with_suffix('.run.json')
        self.ids = list(ids)
        self.fingerprint = signature(identity)
        self.records = {}
        self.prior_api_errors = {}
        self.identity = identity
        self.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive OS-released lock prevents two evaluators from resuming the
        # same output simultaneously; stale process exit cannot strand a lock.
        self.lock_handle = self.output.with_suffix('.lock').open('a+b')
        self.lock_handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            if self.lock_handle.read(1) == b'':
                self.lock_handle.write(b'0'); self.lock_handle.flush()
            self.lock_handle.seek(0)
            msvcrt.locking(self.lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            self._prepare(resume, retry_api_errors)
        except BaseException:
            self.close()
            raise

    def _prepare(self, resume, retry_api_errors):
        existing = self.output.exists() or self.partial.exists() or self.manifest.exists()
        if existing and not resume: raise FileExistsError('Output/run exists; use --resume for exact matching run')
        if existing:
            if not self.manifest.exists():
                raise ValueError('Old output has no run fingerprint; use a new output path, do not merge old predictions')
            meta = json.loads(self.manifest.read_text(encoding='utf-8'))
            if meta['fingerprint'] != self.fingerprint or meta['question_ids'] != self.ids:
                raise ValueError('Resume fingerprint mismatch: model/data/template/code/config changed')
            source = self.output if self.output.exists() else self.partial
            if source.exists():
                raw_lines = source.read_bytes().splitlines(keepends=True)
                for number, raw in enumerate(raw_lines):
                    try: record = json.loads(raw.decode('utf-8'))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        if source != self.partial or number != len(raw_lines)-1 or raw.endswith(b'\n'):
                            raise ValueError('Corrupt completed journal row; refusing resume')
                        # Only an unterminated, malformed final partial row can
                        # be an interrupted write. Preserve bytes and rerun it.
                        tail_archive = self.output.with_suffix('.interrupted_tail.bin')
                        with tail_archive.open('ab') as archive: archive.write(raw)
                        continue
                    ident = record['finqa_id']
                    if ident not in self.ids or record.get('run_fingerprint') != self.fingerprint:
                        raise ValueError('Invalid ID/fingerprint in result journal')
                    if ident in self.records: raise ValueError('Duplicate question in result journal')
                    self.records[ident] = record
            if retry_api_errors:
                self.prior_api_errors = {k:v for k,v in self.records.items() if v['status']=='api_error'}
                if self.prior_api_errors:
                    archive = self.output.with_suffix('.api_error_history.jsonl')
                    with archive.open('a', encoding='utf-8') as h:
                        for record in self.prior_api_errors.values():
                            h.write(json.dumps(record, ensure_ascii=False)+'\n')
                        h.flush(); os.fsync(h.fileno())
                    self.records = {k:v for k,v in self.records.items() if k not in self.prior_api_errors}
            # Rewrite safely to remove retried errors and recover a final output.
            if source.exists():
                temporary = self.partial.with_suffix('.resume.tmp')
                with temporary.open('w', encoding='utf-8') as h:
                    for ident in self.ids:
                        if ident in self.records: h.write(json.dumps(self.records[ident],ensure_ascii=False)+'\n')
                    h.flush(); os.fsync(h.fileno())
                temporary.replace(self.partial)
                if self.output.exists() and len(self.records) < len(self.ids): self.output.unlink()
        else:
            meta = dict(fingerprint=self.fingerprint, question_ids=self.ids, identity=self.identity)
            temp = self.manifest.with_suffix('.tmp')
            temp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
            temp.replace(self.manifest)
        self.handle = self.partial.open('a', encoding='utf-8')

    def pending(self, rows):
        return [r for r in rows if r['finqa_id'] not in self.records]

    def append(self, record):
        ident = record['finqa_id']
        if ident not in self.ids or ident in self.records: raise ValueError('Unknown/duplicate result ID')
        record['run_fingerprint'] = self.fingerprint
        if ident in self.prior_api_errors:
            record['resumed_api_error'] = dict(previous_error=self.prior_api_errors[ident].get('error'))
        self.handle.write(json.dumps(record, ensure_ascii=False)+'\n')
        self.handle.flush(); os.fsync(self.handle.fileno())
        self.records[ident] = record

    def finalize(self):
        if set(self.records) != set(self.ids): raise ValueError('Cannot finalize an incomplete run')
        self.handle.close()
        temp = self.output.with_suffix('.final.tmp')
        with temp.open('w', encoding='utf-8') as h:
            for ident in self.ids: h.write(json.dumps(self.records[ident],ensure_ascii=False)+'\n')
            h.flush(); os.fsync(h.fileno())
        temp.replace(self.output)
        self.partial.unlink(missing_ok=True)

    def close(self):
        if getattr(self,'handle',None) is not None: self.handle.close()
        if getattr(self,'lock_handle',None) is not None: self.lock_handle.close()
