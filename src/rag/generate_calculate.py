"""Task 5: Qwen generation with traceable operands and a safe Decimal calculator.

Only Task 4 rows are read. Gold answers and reference document labels never enter
the generation prompt or the output trace.
"""

from __future__ import annotations

import argparse
import ast
from decimal import Decimal, InvalidOperation, localcontext, ROUND_HALF_UP
import hashlib
import json
import os
from pathlib import Path
import re
import time
from urllib import error, request


PROJECT = Path(__file__).resolve().parents[2]
INPUT = PROJECT / "data/processed/eval/task4_reranked_contexts.jsonl"
OUT_DIR = PROJECT / "data/processed/eval"
MODEL = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")
PROMPT_VERSION = "task5-v1"
MISSING = "缺少相关数据，无法求出答案"
NUMBER = re.compile(r"(?<![0-9A-Za-z_.])[-+]?\d[\d,]*(?:\.\d+)?(?![0-9A-Za-z_.])")
UNITS = {
    "decimal": Decimal(1), "%": Decimal("0.01"), "倍": Decimal(1),
    "元": Decimal(1), "千元": Decimal(1000), "万元": Decimal(10000),
    "百万元": Decimal(1000000), "亿元": Decimal(100000000),
    "股": Decimal(1), "万股": Decimal(10000), "亿股": Decimal(100000000),
    "元/股": Decimal(1), "万元/股": Decimal(10000),
}
UNIT_KIND = {
    **{unit: "ratio" for unit in ("decimal", "%", "倍")},
    **{unit: "money" for unit in ("元", "千元", "万元", "百万元", "亿元")},
    **{unit: "shares" for unit in ("股", "万股", "亿股")},
    **{unit: "per_share" for unit in ("元/股", "万元/股")},
}
SYSTEM = """你是中文财报问答助手。仅使用用户问题和给出的证据，不要猜测缺失数值。
输出一个 JSON 对象，不要 Markdown 或额外文字。模式只有 calculate、direct、insufficient：
calculate: {"mode":"calculate","operands":[{"id":"a","value":"123.45","unit":"万元","source_ref":"E1","quote":"原文中含该数值的完整短语"}],"expression":"a / b - 1","answer_unit":"decimal"}
direct: {"mode":"direct","answer":"简短答案","citations":["E1"]}
insufficient: {"mode":"insufficient","reason":"缺少的具体数据"}
每个计算操作数必须有来源：证据编号 E1 等，或问题本身用 question；quote 必须逐字出现在相应来源。数据来自表格时核对公司、指标、年份和单位；问题假设值优先于报告旧值。金额单位用元、千元、万元、百万元、亿元；比率用 decimal 或 %；股数用股、万股、亿股；每股金额用元/股。表达式只用操作数 id、数字、+ - * / ** 和括号。CAGR 用 `(end / start) ** (1 / years) - 1`。输出比率时 answer_unit=decimal，例如 5.62% 输出 0.0562。不要自己心算，计算由程序执行。"""


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def validate_row(row: dict) -> None:
    forbidden = {"gold", "Reference_documents", "external_knowledge"}
    if forbidden.intersection(row):
        raise ValueError(f"Evaluation field in Task 5 input: {row.get('finqa_id')}")
    if not isinstance(row.get("question"), str) or not isinstance(row.get("selected_evidence"), list):
        raise ValueError("Malformed Task 4 row")
    if len(row.get("context_text", "")) > 2500:
        raise ValueError("Context exceeded Task 4 character budget")


def messages_for(row: dict, mode: str) -> list[dict]:
    validate_row(row)
    if mode not in ("rag", "no_rag"):
        raise ValueError("mode must be rag or no_rag")
    prompt = f"公司：{row['company_with_period']}\n问题：{row['question']}"
    if mode == "rag":
        prompt += "\n\n可用证据（编号用于引用）：\n" + row["context_text"]
    else:
        prompt += "\n\n未提供外部文档；只能使用问题中明确给出的数值。"
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt}]


def _source_text(row: dict, source_ref: str, mode: str) -> str:
    if source_ref == "question":
        return row["question"]
    if mode != "rag":
        raise ValueError("No external evidence in no_rag mode")
    for evidence in row["selected_evidence"]:
        if evidence["citation"] == source_ref:
            pattern = re.compile(r"(?m)^\[" + re.escape(source_ref) + r"\].*?(?=^\[E\d+\]|\Z)", re.S)
            match = pattern.search(row["context_text"])
            if not match:
                raise ValueError(f"Citation absent from context: {source_ref}")
            return match.group()
    raise ValueError(f"Unknown citation: {source_ref}")


def _operand_value(item: dict, row: dict, mode: str) -> Decimal:
    value_text = str(item["value"]).replace(",", "")
    value = Decimal(value_text)
    if not value.is_finite():
        raise ValueError("Nonfinite operand")
    unit = item["unit"]
    if unit not in UNITS:
        raise ValueError(f"Unsupported unit: {unit}")
    quote = item["quote"]
    source = _source_text(row, item["source_ref"], mode)
    if not isinstance(quote, str) or not quote or quote not in source:
        raise ValueError("Operand quote absent from claimed source")
    quoted_values = [Decimal(x.group().replace(",", "")) for x in NUMBER.finditer(quote)]
    if value not in quoted_values:
        raise ValueError("Operand value absent from quote")
    return value * UNITS[unit]


def _calculate(expression: str, operands: dict[str, Decimal]) -> Decimal:
    if len(expression) > 300:
        raise ValueError("Expression too long")
    tree = ast.parse(expression, mode="eval")

    def walk(node: ast.AST, depth: int = 0) -> Decimal:
        if depth > 20:
            raise ValueError("Expression too deep")
        if isinstance(node, ast.Expression):
            return walk(node.body, depth + 1)
        if isinstance(node, ast.Name) and node.id in operands:
            return operands[node.id]
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return Decimal(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = walk(node.operand, depth + 1)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            left, right = walk(node.left, depth + 1), walk(node.right, depth + 1)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if abs(right) > 12:
                raise ValueError("Exponent too large")
            return left ** right
        raise ValueError("Unsupported expression syntax")

    with localcontext() as ctx:
        ctx.prec = 40
        result = walk(tree)
        if not result.is_finite() or abs(result) > Decimal("1e30"):
            raise ValueError("Result out of bounds")
        return +result


def process_response(raw: str, row: dict, mode: str) -> dict:
    """Return canonical answer separately from an auditable calculation trace."""
    result = {"status": "invalid_plan", "final": {"answer": None}, "trace": {}, "error": None}
    try:
        plan = json.loads(raw)
        if not isinstance(plan, dict):
            raise ValueError("Response is not a JSON object")
        kind = plan.get("mode")
        if kind == "insufficient":
            result.update(status="insufficient", final={"answer": MISSING},
                          trace={"reason": plan.get("reason", "")})
        elif kind == "direct":
            refs = plan.get("citations", [])
            if not isinstance(refs, list):
                raise ValueError("citations must be a list")
            for ref in refs:
                _source_text(row, ref, mode)
            answer = plan.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Direct answer must be nonempty text")
            result.update(status="direct", final={"answer": answer.strip()},
                          trace={"citations": refs})
        elif kind == "calculate":
            items = plan.get("operands")
            if not isinstance(items, list) or not 1 <= len(items) <= 12:
                raise ValueError("Expected 1–12 operands")
            values = {}
            for item in items:
                ident = item["id"]
                if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", ident) or ident in values:
                    raise ValueError("Invalid or duplicate operand id")
                values[ident] = _operand_value(item, row, mode)
            unit = plan.get("answer_unit", "decimal")
            if unit not in UNITS:
                raise ValueError("Unsupported answer unit")
            expression = plan["expression"]
            if not isinstance(expression, str):
                raise ValueError("Expression must be text")
            numeric = _calculate(expression, values) / UNITS[unit]
            with localcontext() as ctx:
                ctx.prec = 40
                rounded = numeric.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            answer = format(rounded, "f").rstrip("0").rstrip(".") if "." in format(rounded, "f") else format(rounded, "f")
            if answer in ("", "-0"):
                answer = "0"
            result.update(status="calculated", final={"answer": answer},
                          trace={"operands": items, "normalized_base_values":
                                 {key: str(value) for key, value in values.items()},
                                 "expression": expression, "answer_unit": unit,
                                 "unrounded": str(numeric)})
        else:
            raise ValueError("Unknown response mode")
    except (ValueError, KeyError, TypeError, InvalidOperation, ZeroDivisionError,
            OverflowError, SyntaxError) as exc:
        result["error"] = str(exc)
    return result


def call_model(endpoint: str, messages: list[dict], timeout: int) -> tuple[str, dict]:
    url = endpoint.rstrip("/") + "/chat/completions"
    payload = {"model": MODEL, "messages": messages, "temperature": 0,
               "max_tokens": 512, "response_format": {"type": "json_object"}}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url, body, {"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=timeout) as response:
            data = json.load(response)
    except error.HTTPError as exc:
        raise RuntimeError(f"Model HTTP {exc.code}: {exc.read(500).decode('utf-8', 'replace')}") from exc
    return data["choices"][0]["message"]["content"], data.get("usage", {})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("rag", "no_rag"), required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    rows = list(read_jsonl(INPUT))
    for row in rows:
        validate_row(row)
    if len({row["finqa_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate question IDs")
    rows = rows[:args.limit]
    if args.dry_run:
        print(json.dumps({"questions": len(rows), "mode": args.mode,
                          "model": MODEL, "prompt_version": PROMPT_VERSION,
                          "example": messages_for(rows[0], args.mode)}, ensure_ascii=False, indent=2))
        return
    suffix = f"_smoke{args.limit}" if args.limit else ""
    output = OUT_DIR / f"task5_{args.mode}{suffix}_generation.jsonl"
    if output.exists():
        raise FileExistsError(f"Existing result is protected: {output}")
    started = time.time()
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            messages = messages_for(row, args.mode)
            response_started = time.perf_counter()
            try:
                raw, usage = call_model(args.endpoint, messages, args.timeout)
                processed = process_response(raw, row, args.mode)
                failure = None
            except (RuntimeError, TimeoutError, error.URLError, KeyError) as exc:
                raw, usage, failure = None, {}, str(exc)
                processed = {"status": "api_error", "final": {"answer": None},
                             "trace": {}, "error": failure}
            record = {"finqa_id": row["finqa_id"], "mode": args.mode,
                      "model": MODEL, "prompt_version": PROMPT_VERSION,
                      "prompt_sha256": hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest(),
                      "source_chunk_ids": [item["chunk_id"] for item in row["selected_evidence"]] if args.mode == "rag" else [],
                      "raw_response": raw, "usage": usage,
                      "latency_seconds": round(time.perf_counter() - response_started, 3),
                      **processed}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{index}/{len(rows)} {row['finqa_id']} {processed['status']}", flush=True)
    print(json.dumps({"output": str(output), "rows": len(rows),
                      "elapsed_seconds": round(time.time() - started, 2)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
