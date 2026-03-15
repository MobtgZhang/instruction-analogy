"""
Part 1：根据类比数据与在线百科信息，合成 LLM 训练指令 (input, thinking, output)。
读取类比数据由 common.load_analogy_data 在 run.py 中调用，本模块只负责单条合成。
"""
import json

from common import get_prompts, parse_json_from_llm


def synthesize_instruction(record: dict, encyclopedia: str, llm_generate) -> dict | None:
    """
    根据单条类比记录与百科信息，合成一条 (input, thinking, output)。
    record 需包含：analogy_a, analogy_b, attr_a, attr_b, relation, order 等。
    llm_generate(prompt, max_tokens) -> str。
    成功返回 {"input", "thinking", "output"}，失败返回 None。
    """
    prompts = get_prompts()
    template = prompts.get("synthesis_prompt", "").strip()
    if not template:
        return None

    prompt = template.format(
        analogy_a=record.get("analogy_a", ""),
        analogy_b=record.get("analogy_b", ""),
        attr_a=record.get("attr_a", ""),
        attr_b=record.get("attr_b", ""),
        relation=record.get("relation", ""),
        order=record.get("order", "1st-order"),
        encyclopedia=encyclopedia or "（无）",
    )
    raw = llm_generate(prompt, max_tokens=2048)
    try:
        obj = parse_json_from_llm(raw)
        for key in ("input", "thinking", "output"):
            if key not in obj or not str(obj[key]).strip():
                return None
        return {
            "input": str(obj["input"]).strip(),
            "thinking": str(obj["thinking"]).strip(),
            "output": str(obj["output"]).strip(),
        }
    except (json.JSONDecodeError, KeyError):
        return None
