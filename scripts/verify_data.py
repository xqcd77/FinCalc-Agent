import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data/manifests"
EXPECTED = {
    "corporate_reports": (163, 141, 141),
}


def load_jsonl(name):
    return [json.loads(line) for line in (MANIFESTS / name).read_text(encoding="utf-8").splitlines() if line.strip()]


questions = load_jsonl("questions.jsonl")
question_files = load_jsonl("question_files.jsonl")
latex_mapping = load_jsonl("latex_mapping.jsonl")
latex_files = load_jsonl("latex_files.jsonl")
missing = json.loads((MANIFESTS / "latex_missing.json").read_text(encoding="utf-8"))

assert len(questions) == 163
assert len(question_files) == 141
assert len(latex_mapping) == 141
assert len(latex_files) == 141
assert not missing
assert len({q["finqa_id"] for q in questions}) == 163

for record in question_files + latex_files:
    path = ROOT / record["local_path"]
    assert path.is_file(), path
    raw = path.read_bytes()
    assert len(raw) == record["bytes"], path
    assert hashlib.sha256(raw).hexdigest() == record["sha256"], path

ref_keys = {(x["scene"], x["reference"]) for x in latex_mapping}
assert all((q["scene"], q["reference_document"]) in ref_keys for q in questions)

rows = []
for scene, (nq, nfiles, ntex) in EXPECTED.items():
    result = {
        "scene": scene,
        "questions": sum(q["local_scene"] == scene for q in questions),
        "question_files": sum(Path(x["local_path"]).parts[4] == scene for x in question_files),
        "latex_files": sum(x["local_scene"] == scene for x in latex_mapping),
    }
    assert (result["questions"], result["question_files"], result["latex_files"]) == (nq, nfiles, ntex), result
    rows.append(result)

all_dirs = [ROOT] + [p for p in ROOT.rglob("*") if p.is_dir()]
assert all((folder / "README.md").is_file() for folder in all_dirs)
pdfs = list(ROOT.rglob("*.pdf"))
assert not pdfs, pdfs[:5]

summary = {
    "source": "https://huggingface.co/datasets/Multilingual-Multimodal-NLP/FinanceComplexQA",
    "download_mirror": "https://hf-mirror.com/datasets/Multilingual-Multimodal-NLP/FinanceComplexQA",
    "repository_head_at_download": "77d63e305d522d4db23f92d4e062de1bfde9eb07",
    "question_archive_sha256": "b8a7c93f7cb53adfe2a715b953af1e3a48fa557a9f47814c1854df960d5ca616",
    "reference_archive_sha256_expected": "945df272e97c4a2f7389d527997fbf3e8aaf7c4066e85145dda3b381cb213cbe",
    "archive_method": "HTTP byte ranges; selected .tex entries only; ZIP member CRC verified during extraction",
    "total_questions": len(questions),
    "total_question_files": len(question_files),
    "total_latex_files": len(latex_files),
    "pdf_files": len(pdfs),
    "scenes": rows,
}
(MANIFESTS / "download_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

report = [
    "# 数据下载与校验结果",
    "",
    "数据来源：[FinanceComplexQA 官方数据页](https://huggingface.co/datasets/Multilingual-Multimodal-NLP/FinanceComplexQA)。当前下载了 `FinComplexQA-Pro/CN/scene_categories` 的公司财报题目，以及题目 `Reference_documents` 指向的 LaTeX 原件。未下载 PDF，也未下载英文题、`overall` 或重复的 task_categories 视图。",
    "",
    "| 领域 | 题数 | 原始题目文件 | 对应 LaTeX 文件 |",
    "|---|---:|---:|---:|",
]
names = {
    "corporate_reports": "公司财报",
}
for row in rows:
    report.append(f"| {names[row['scene']]} | {row['questions']} | {row['question_files']} | {row['latex_files']} |")
report += [
    f"| 合计 | {len(questions)} | {len(question_files)} | {len(latex_files)} |",
    "",
    "公司财报题目以 `finqa_id` 去重，163 个 ID 全部唯一；141 个不同的文档引用均匹配到 `.tex`。题目文件和 LaTeX 文件的大小、SHA-256 已对照清单复核，所有 ZIP 成员在读取时通过 CRC 校验。项目内 PDF 数量为 0。",
    "",
    "注意：这里的 `latex_files` 是**被题目实际引用**的文件数；官方某领域目录里的全部 LaTeX 文件可能更多。FinanceComplexQA 是保留评测集，不用于 LoRA 或 GRPO 训练。",
    "",
    "清单：`questions.jsonl` 为逐题引用关系；`question_files.jsonl` 与 `latex_files.jsonl` 为逐文件校验记录；`latex_mapping.jsonl` 为原始引用路径到 ZIP 成员的映射；`download_summary.json` 为机器可读汇总。",
]
(ROOT / "docs/DATASET.md").write_text("\n".join(report) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))

