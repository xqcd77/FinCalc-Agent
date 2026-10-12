# 运行指南

## 1. CPU环境与检查

在仓库根目录安装`requirements.txt`，运行`examples/decimal_demo.py`和`unittest discover`。
CUDA训练／服务建议使用Linux环境。PyTorch根据实际CUDA驱动安装；记录的参考环境为PyTorch2.8.0、Transformers5.2.0、PEFT0.18.1。
当前代码使用Qwen3.5原生类，不能随意换成旧版Transformers。

## 2. 数据和检索模型

```bash
python scripts/download_selected.py
python scripts/verify_data.py
python -m pip install -r requirements/retrieval.txt
python scripts/download_bge_m3.py
python scripts/download_bge_reranker_large.py
python src/rag/parse_latex.py
python src/rag/chunk_evidence.py
python src/rag/hybrid_retrieval.py all
```

下载脚本从官方来源获取选定语料并核对记录的校验值；上游版本变化时应停止并检查，不绕过校验。
模型文件与索引均被.gitignore排除。首次下载耗时与体积可能较大。

下载Qwen3.5-9B到`models/Qwen3.5-9B`，然后导出R1上下文：

```bash
python scripts/compare_financial_rerank_v49.py --output reports/local/rerank --coverage-guard --skip-comparison --export-reranked data/processed/eval/r1_rankings.jsonl --export-contexts data/processed/eval/r1_contexts.jsonl --tokenizer-path models/Qwen3.5-9B/tokenizer.json
```

`--skip-comparison`仅跳过依赖本地标注的计分，候选打分、覆盖保护和上下文导出保持相同。

## 3. LoRA

```bash
python -m pip install -r requirements/training.txt
python scripts/train_lora.py --profile configs/lora.json --preflight-only
python scripts/train_lora.py --profile configs/lora.json
```

完整5,016条数据没有随源码发布。先将自己拥有使用权、已审查的训练数据按[数据契约](DATA.md)放入相应目录，并调整配置计数与版本。
每个split须有prompt/completion和metadata，manifest的文件hash必须匹配，verified计数一致。
缺少这些文件时训练入口会拒绝启动。示例不满足整个训练发布契约。

## 4. BF16服务

Linux示例（PowerShell使用`$env:变量名='值'`设置相同环境变量）：

```bash
export QWEN_MODEL_PATH=models/Qwen3.5-9B
export PEFT_ADAPTER_PATH=experiments/lora_r32_e3/best_adapter
export QWEN_MODEL_ID=Qwen3.5-9B-LoRA-BF16
export QWEN_CONTEXT_LIMIT=3200
python scripts/serve_qwen35_9b_lora_bf16.py
```

GPU须支持BF16；没有权重或CUDA时会明确报错。服务默认端口8000。

## 5. 生成、续评与离线计分

```bash
python scripts/run_rag.py --input data/processed/eval/r1_contexts.jsonl --output outputs/rag.jsonl --tokenizer-path models/Qwen3.5-9B/tokenizer.json --model Qwen3.5-9B-LoRA-BF16 --precision-label bf16 --question-workers 2 --skip-evaluation
python scripts/evaluate.py --input outputs/rag.jsonl --contexts data/processed/eval/r1_contexts.jsonl --output-prefix reports/local/evaluation
```

中断后给同一生成命令增加`--resume`；仅对明确API失败使用`--retry-api-errors`。输入／模型／配置指纹变化时拒绝混合续评。
使用完整官方数据准备默认gold目录后才能计算答案准确率；检索证据指标另需独立qrels。
`--help`可查看入口参数，不会发出模型请求。
