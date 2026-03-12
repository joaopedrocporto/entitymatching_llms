from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _extract_text_from_openai_record(record: Dict[str, Any]) -> str:
    resp = record.get("response") or {}
    body = resp.get("body") if isinstance(resp, dict) else None
    if not body:
        return ""

    choices = body.get("choices")
    if isinstance(choices, list) and len(choices) > 0:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        t = block.get("text")
                        if t:
                            parts.append(t)
                if parts:
                    return "\n".join(parts).strip()

    output = body.get("output")
    if not isinstance(output, list):
        return ""
    texts: List[str] = []
    for item in output:
        if item.get("type") != "message":
            continue
        content = item.get("content") or []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "output_text":
                t = block.get("text")
                if t:
                    texts.append(t)
    return "\n".join(texts).strip()


def _parse_yes_no_from_text(text: str) -> bool:
    if not text:
        return False
    lower = text.strip().lower()
    for line in lower.splitlines()[::-1]:
        line = line.strip()
        if line.startswith("answer:"):
            if "yes" in line:
                return True
            if "no" in line:
                return False
    if "answer: yes" in lower or lower.endswith("yes"):
        return True
    if "answer: no" in lower or lower.endswith("no"):
        return False
    snippet = text.strip().replace("\n", " ")
    snippet = snippet[:120] + "..." if len(snippet) > 120 else snippet
    print(f"[AVISO] Não foi possível extrair YES/NO automaticamente. Revisar manualmente. Trecho: {snippet}")
    return False


def load_predictions_from_openai_jsonl(path: str | Path) -> Dict[int, bool]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    preds: Dict[int, bool] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            custom_id = record.get("custom_id")
            if not isinstance(custom_id, str):
                continue
            try:
                idx = int(custom_id.split("-")[-1])
            except ValueError:
                continue
            text = _extract_text_from_openai_record(record)
            pred = _parse_yes_no_from_text(text)
            preds[idx] = pred
    return preds


def load_predictions_from_llama_jsonl(path: str | Path) -> Dict[int, bool]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    preds: Dict[int, bool] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            custom_id = record.get("custom_id")
            label_str = (record.get("label") or "").strip().upper()

            if not isinstance(custom_id, str):
                continue

            try:
                idx = int(custom_id.split("-")[-1])
            except ValueError:
                continue

            if label_str == "YES":
                preds[idx] = True
            elif label_str == "NO":
                preds[idx] = False
            else:
                raw = record.get("raw_response", "")
                if raw:
                    preds[idx] = _parse_yes_no_from_raw(raw)
                else:
                    continue

    return preds


def load_predictions_from_mistral_jsonl(path: str | Path) -> Tuple[Dict[int, bool], Dict[int, bool]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    preds: Dict[int, bool] = {}
    ground_truth: Dict[int, bool] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            idx = record.get("index")
            if idx is None:
                continue

            pred_val = record.get("prediction")
            gt_val = record.get("ground_truth")

            if isinstance(pred_val, bool):
                preds[idx] = pred_val
            elif isinstance(pred_val, str):
                preds[idx] = pred_val.upper() in {"YES", "TRUE", "1"}

            if isinstance(gt_val, bool):
                ground_truth[idx] = gt_val
            elif isinstance(gt_val, (int, str)):
                ground_truth[idx] = bool(gt_val) if isinstance(gt_val, int) else gt_val.upper() in {"YES", "TRUE", "1"}

    return preds, ground_truth


def _parse_yes_no_from_raw(raw: str) -> bool:
    if not raw:
        return False
    lower = raw.strip().lower()
    if "answer: yes" in lower or lower.endswith("yes") or "decision: yes" in lower:
        return True
    if "answer: no" in lower or lower.endswith("no") or "decision: no" in lower:
        return False
    snippet = raw.strip().replace("\n", " ")
    snippet = snippet[:120] + "..." if len(snippet) > 120 else snippet
    print(f"[AVISO] Não foi possível extrair YES/NO de raw_response. Revisar manualmente. Trecho: {snippet}")
    return False


def _extract_text_from_qwen_record(record: Dict[str, Any]) -> str:
    resp = record.get("response") or {}
    body = resp.get("body") if isinstance(resp, dict) else None
    if not body:
        return ""
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        return ""
    msg = choices[0].get("message")
    if not isinstance(msg, Dict):
        return ""
    content = msg.get("content")
    if content is None:
        return ""
    return str(content).strip()


def load_predictions_from_qwen_jsonl(path: str | Path) -> Dict[int, bool]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    preds: Dict[int, bool] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            custom_id = record.get("custom_id")
            if not isinstance(custom_id, str):
                continue
            try:
                idx = int(custom_id.split("-")[-1])
            except ValueError:
                continue
            text = _extract_text_from_qwen_record(record)
            preds[idx] = _parse_yes_no_from_text(text)
    return preds

if __name__ == "__main__":
    path = Path("results/amazon-walmart/llama3/prompt1.jsonl")
    preds = load_predictions_from_qwen_jsonl(path)

