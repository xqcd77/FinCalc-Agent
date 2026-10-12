"""Download the pinned BAAI/bge-reranker-large model snapshot.

Files are fetched from the Hugging Face mirror with resumable byte ranges.
Model weights and tokenizer/config files are SHA-256 verified before use.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import ssl
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models/reranker/bge-reranker-large"
REVISION = "55611d7bca2a7133960a6d3b71e083071bbfc312"
BASE_URL = f"https://hf-mirror.com/BAAI/bge-reranker-large/resolve/{REVISION}/"
WEIGHT_NAME = "model.safetensors"
WEIGHT_SIZE = 2_239_618_772
WEIGHT_SHA256 = "c5ae4e262c60ed2fb507ec587358285377ec36ee0a2e6da029f0534272b06d36"
FILES = {
    "config.json": (801, "25dda0319f57e9de243c32eb7ecefa0768271afac6b5d8a94ba7f779198ab102"),
    "sentencepiece.bpe.model": (5_069_051, "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865"),
    "special_tokens_map.json": (279, "d5469a60db23249c7f8945013d78df30b44b6bf686c6bb4740f4223f77b1b535"),
    "tokenizer.json": (17_098_107, "9eb652ac4e40cc093272bbbe0f55d521cf67570060227109b5cdc20945a4489e"),
    "tokenizer_config.json": (443, "a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b"),
}
PART_SIZE = 4 * 1024 * 1024
SSL_CONTEXT = ssl.create_default_context()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def fetch_part(part: int, parts_dir: Path) -> Path:
    start = part * PART_SIZE
    end = min(WEIGHT_SIZE, start + PART_SIZE) - 1
    output = parts_dir / f"{part:04d}.part"
    expected_size = end - start + 1
    if output.is_file() and output.stat().st_size == expected_size:
        return output
    for attempt in range(5):
        try:
            request = urllib.request.Request(
                BASE_URL + WEIGHT_NAME,
                headers={"Range": f"bytes={start}-{end}",
                         "User-Agent": "Mozilla/5.0 finance-rag-model-fetch"})
            with urllib.request.urlopen(request, timeout=120,
                                        context=SSL_CONTEXT) as response:
                expected_range = f"bytes {start}-{end}/"
                if response.status != 206 or not response.headers.get(
                        "Content-Range", "").startswith(expected_range):
                    raise RuntimeError("Mirror returned an invalid byte range")
                with output.open("wb") as target:
                    while block := response.read(1024 * 1024):
                        target.write(block)
            if output.stat().st_size != expected_size:
                raise RuntimeError("Downloaded range has an invalid size")
            return output
        except Exception:
            output.unlink(missing_ok=True)
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Unreachable")


def download_weight() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    weight = DEST / WEIGHT_NAME
    if weight.is_file() and weight.stat().st_size == WEIGHT_SIZE \
            and sha256(weight) == WEIGHT_SHA256:
        print("Pinned reranker weight already verified", flush=True)
        return

    parts_dir = DEST / ".model_parts"
    parts_dir.mkdir(exist_ok=True)
    count = (WEIGHT_SIZE + PART_SIZE - 1) // PART_SIZE
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch_part, part, parts_dir) for part in range(count)]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed % 10 == 0 or completed == count:
                print(f"Weight ranges: {completed}/{count}", flush=True)

    partial = DEST / f"{WEIGHT_NAME}.partial"
    digest = hashlib.sha256()
    with partial.open("wb") as target:
        for part in range(count):
            with (parts_dir / f"{part:04d}.part").open("rb") as source:
                while block := source.read(1024 * 1024):
                    digest.update(block)
                    target.write(block)
    if partial.stat().st_size != WEIGHT_SIZE or digest.hexdigest() != WEIGHT_SHA256:
        partial.unlink(missing_ok=True)
        raise ValueError(f"Model weight checksum mismatch: {digest.hexdigest()}")
    partial.replace(weight)
    for part in parts_dir.glob("*.part"):
        part.unlink()
    parts_dir.rmdir()
    print(f"Verified {WEIGHT_NAME}: {WEIGHT_SHA256}", flush=True)


def download_small_files() -> None:
    for name, (size, expected_sha256) in FILES.items():
        path = DEST / name
        if path.is_file() and path.stat().st_size == size \
                and sha256(path) == expected_sha256:
            continue
        request = urllib.request.Request(
            BASE_URL + name,
            headers={"User-Agent": "Mozilla/5.0 finance-rag-model-fetch"})
        with urllib.request.urlopen(request, timeout=90,
                                   context=SSL_CONTEXT) as response:
            data = response.read()
        path.write_bytes(data)
        if len(data) != size or sha256(path) != expected_sha256:
            path.unlink(missing_ok=True)
            raise ValueError(f"File SHA-256 verification failed: {name}")
        print(f"Verified {name}: {size} bytes", flush=True)


if __name__ == "__main__":
    download_weight()
    download_small_files()
