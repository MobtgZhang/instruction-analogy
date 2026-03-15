"""
Part 2：验证指令是否正确。
根据原始类比依据与训练样本 (input, thinking, output)，判断事实一致、推理链完整、表达合理。
"""
import json

from common import get_prompts, parse_json_from_llm


def validate_instruction(
    instruction: dict,
    record: dict,
    llm_generate,
) -> tuple[bool, list[str]]:
    """
    验证指令是否正确。
    instruction: {"input", "thinking", "output"}
    record: 原始类比记录，需含 analogy_a, analogy_b, relation。
    llm_generate(prompt, max_tokens) -> str。
    返回 (是否通过, 问题列表)。
    """
    prompts = get_prompts()
    template = prompts.get("validate_prompt", "").strip()
    if not template:
        return False, ["未配置 validate_prompt"]

    prompt = template.format(
        input=instruction.get("input", ""),
        thinking=instruction.get("thinking", ""),
        output=instruction.get("output", ""),
        analogy_a=record.get("analogy_a", ""),
        analogy_b=record.get("analogy_b", ""),
        relation=record.get("relation", ""),
    )
    raw = llm_generate(prompt, max_tokens=512)
    try:
        obj = parse_json_from_llm(raw)
        valid = obj.get("valid", False)
        issues = obj.get("issues") or []
        if not isinstance(issues, list):
            issues = [str(issues)]
        return bool(valid), list(issues)
    except (json.JSONDecodeError, KeyError):
        return False, ["验证输出解析失败"]
