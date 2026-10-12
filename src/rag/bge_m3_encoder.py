"""Official BAAI BGE-M3 ONNX dense encoder; no query prefix or silent truncation."""
import hashlib
import json
from pathlib import Path
import numpy as np

MODEL = 'BAAI/bge-m3'
DIMENSION = 1024
MAX_LENGTH = 2048
BATCH_SIZE = 2

def model_fingerprint(path):
    manifest = json.loads((Path(path) / 'download_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('model') != MODEL: raise ValueError('Wrong embedding model manifest')
    for item in manifest['files']:
        p = Path(path) / item['file']
        if not p.is_file() or p.stat().st_size != item['bytes']: raise ValueError('Incomplete BGE-M3 download')
        digest = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(4 * 1024 * 1024), b''): digest.update(block)
        if digest.hexdigest() != item['sha256']: raise ValueError('BGE-M3 file checksum changed: '+p.name)
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()

class BgeM3Encoder:
    def __init__(self, path, max_length=MAX_LENGTH, threads=4):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.path = Path(path)
        self.max_length = max_length
        if not 2 <= max_length <= 8192: raise ValueError('Invalid BGE-M3 sequence limit')
        self.tokenizer = Tokenizer.from_file(str(self.path / 'tokenizer.json'))
        self.tokenizer.no_truncation(); self.tokenizer.no_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads; options.inter_op_num_threads = 1
        # Keep CPU memory bounded on the local 16GB machine: do not keep a
        # second prepacked copy of the 2.27GB float32 weights.
        options.add_session_config_entry('session.disable_prepacking','1')
        self.session = ort.InferenceSession(str(self.path / 'model.onnx'), sess_options=options,
            providers=['CPUExecutionProvider'])
        self.inputs = {x.name for x in self.session.get_inputs()}
        self.outputs = [x.name for x in self.session.get_outputs()]
        self.output_name=next((n for n in ('sentence_embedding','last_hidden_state','token_embeddings') if n in self.outputs),None)
        if self.output_name is None:raise ValueError('Official dense output not found')
        config = json.loads((self.path / 'config.json').read_text(encoding='utf-8'))
        self.pad_id = config.get('pad_token_id', 1)

    def embed(self, texts, batch_size=BATCH_SIZE):
        texts = list(texts)
        for start in range(0, len(texts), batch_size):
            enc = self.tokenizer.encode_batch(texts[start:start+batch_size])
            if any(len(x.ids) > self.max_length for x in enc):
                raise ValueError('BGE-M3 input exceeds token limit; preserve evidence and increase limit explicitly')
            width = max(len(x.ids) for x in enc)
            ids = np.full((len(enc), width), self.pad_id, dtype=np.int64)
            mask = np.zeros_like(ids)
            for i,x in enumerate(enc): ids[i,:len(x.ids)] = x.ids; mask[i,:len(x.ids)] = 1
            feed = {'input_ids': ids, 'attention_mask': mask}
            if 'token_type_ids' in self.inputs: feed['token_type_ids'] = np.zeros_like(ids)
            # Official export supplies token_embeddings and sentence_embedding.
            # Request its prepooled dense output explicitly; do not rely on order.
            output = self.session.run([self.output_name], {k:v for k,v in feed.items() if k in self.inputs})[0]
            dense = output[:,0,:] if output.ndim == 3 else output
            if dense.shape != (len(enc), DIMENSION) or not np.isfinite(dense).all():
                raise ValueError('Unexpected BGE-M3 output shape/values: '+str(dense.shape))
            dense = dense.astype(np.float32)
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            if (norms < 1e-12).any(): raise ValueError('Zero embedding')
            yield from dense / norms
