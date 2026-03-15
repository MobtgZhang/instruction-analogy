"""
Part 3：当验证不通过时，根据验证反馈与原始类比依据对指令进行修改。
"""
import json

from common import get_prompts, parse_json_from_llm


def correct_instruction(
    instruction: dict,
    record: dict,
    issues: list[str],
    llm_generate,
) -> dict | None:
    """
    根据验证反馈修正指令。
    instruction: 当前 { input, thinking, output }
    record: 原始类比记录
    issues: 验证返回的问题列表
    llm_generate(prompt, max_tokens) -> str。
    返回修正后的 {"input", "thinking", "output"} 或 None。
    """
    prompts = get_prompts()
    template = prompts.get("correction_prompt", "").strip()
    if not template:
        return None

    prompt = template.format(
        input=instruction.get("input", ""),
        thinking=instruction.get("thinking", ""),
        output=instruction.get("output", ""),
        analogy_a=record.get("analogy_a", ""),
        analogy_b=record.get("analogy_b", ""),
        relation=record.get("relation", ""),
        issues="\n".join(issues),
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
