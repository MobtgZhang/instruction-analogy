"""
指令合成总控脚本：读取类比数据 → 合成指令 (input, thinking, output) → 验证 → 不正确则修改 → 写入。
配置与 prompt 见 config 目录（api.json、*_prompt.txt）。
"""
import os
import json
import argparse
from pathlib import Path

from common import (
    load_config,
    get_prompts,
    get_llm_generator,
    load_analogy_data,
    fetch_encyclopedia_info,
    parse_json_from_llm,
)
from synthesize import synthesize_instruction
from validate import validate_instruction
from correct import correct_instruction

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent

BLOOM_LEVELS = ("Factual", "Conceptual")
BLOOM_FACTUAL_TYPES = [
    "实体类比", "属性类比", "时间类比", "空间类比", "量化类比", "事件类比",
]


def _classify_instruction(instruction: dict, llm_generate) -> dict:
    """对指令做 Bloom 分类，返回 { bloom_level, bloom_types }。"""
    prompts = get_prompts()
    template = prompts.get("classify_prompt", "").strip()
    if not template:
        return {"bloom_level": "Factual", "bloom_types": []}
    prompt = template.format(
        input=instruction.get("input", ""),
        thinking=instruction.get("thinking", ""),
        output=instruction.get("output", ""),
    )
    raw = llm_generate(prompt, max_tokens=512)
    try:
        obj = parse_json_from_llm(raw)
        level = (obj.get("bloom_level") or "").strip()
        types = obj.get("bloom_types") or []
        if isinstance(types, str):
            types = [t.strip() for t in types.split(",") if t.strip()]
        if level not in BLOOM_LEVELS:
            level = "Factual" if any(t in BLOOM_FACTUAL_TYPES for t in types) else "Conceptual"
        return {"bloom_level": level, "bloom_types": list(types)}
    except (json.JSONDecodeError, KeyError):
        return {"bloom_level": "Factual", "bloom_types": []}


def run(
    data_dir: str,
    out_path: str,
    llm_model: str = None,
    api_key: str = None,
    base_url: str = None,
    use_encyclopedia: bool = True,
    max_instructions: int = 0,
    max_correction_retries: int = 2,
):
    """
    主流程：
    1. 读取类比数据
    2. 对每条：合成 (input, thinking, output) → 分类 → 验证
    3. 若验证不通过则修正，直至通过或达到重试上限
    4. 通过的指令追加写入 out_path
    """
    cfg = load_config()
    llm = get_llm_generator(
        model_name=llm_model or cfg.get("llm_model"),
        api_key=api_key or cfg.get("openai_api_key"),
        base_url=base_url or cfg.get("openai_base_url"),
    )
    if not llm:
        print("未配置 API 或 LLM 初始化失败，请检查 config/api.json 或 OPENAI_API_KEY。")
        return

    records = load_analogy_data(data_dir)
    if not records:
        print(f"未在 {data_dir} 下找到类比数据。")
        return

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    written = 0
    print(f"共加载 {len(records)} 条类比数据，开始合成 → 验证 → 修正 …")

    for i, rec in enumerate(records):
        if max_instructions and written >= max_instructions:
            break

        # 1. 百科信息（可选）
        encyclopedia = ""
        if use_encyclopedia and (rec.get("analogy_a") or rec.get("analogy_b")):
            encyclopedia = fetch_encyclopedia_info(
                rec.get("analogy_a", ""), rec.get("analogy_b", ""), max_results=2
            )

        # 2. 合成指令 (input, thinking, output)
        instruction = synthesize_instruction(rec, encyclopedia, llm)
        if not instruction:
            print(f"  [{i+1}] 合成失败，跳过")
            continue

        # 分类（Bloom）
        classification = _classify_instruction(instruction, llm)
        instruction["bloom_level"] = classification["bloom_level"]
        instruction["bloom_types"] = classification["bloom_types"]

        # 3. 验证指令是否正确
        valid, issues = validate_instruction(instruction, rec, llm)

        # 4. 若不正确则修改，再验证
        retries = 0
        while not valid and retries < max_correction_retries:
            corrected = correct_instruction(instruction, rec, issues, llm)
            if not corrected:
                break
            instruction = corrected
            instruction["bloom_level"] = classification["bloom_level"]
            instruction["bloom_types"] = classification["bloom_types"]
            valid, issues = validate_instruction(instruction, rec, llm)
            retries += 1

        if not valid:
            print(f"  [{i+1}] 验证未通过且修正达上限，跳过")
            continue

        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(instruction, ensure_ascii=False) + "\n")
        written += 1
        if written % 10 == 0:
            print(f"  已写入 {written} 条")

    print(f"完成。共写入 {written} 条指令至 {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="类比推理指令合成：读类比数据 → 合成指令 → 验证 → 不正确则修改"
    )
    parser.add_argument("--data-dir", type=str, default=None,
                        help="类比数据目录（含 entity/relation/subgraph jsonl 或 all_analogy.jsonl）")
    parser.add_argument("--out", type=str, default=None, help="输出指令 jsonl 路径")
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--no-encyclopedia", action="store_true", help="不使用在线百科检索")
    parser.add_argument("--max-instructions", type=int, default=0, help="最多合成条数，0 表示不限制")
    parser.add_argument("--max-correction-retries", type=int, default=2)
    args = parser.parse_args()

    cfg = load_config()
    data_dir = args.data_dir or cfg.get("data_dir") or os.path.join(PROJECT_ROOT, "AnalogyKG", "data", "balanced")
    out_path = args.out or cfg.get("out_path") or os.path.join(ROOT_DIR, "data", "instructions.jsonl")

    run(
        data_dir=data_dir,
        out_path=out_path,
        llm_model=args.llm_model,
        api_key=args.api_key,
        base_url=args.base_url,
        use_encyclopedia=not args.no_encyclopedia,
        max_instructions=args.max_instructions or 0,
        max_correction_retries=args.max_correction_retries,
    )


if __name__ == "__main__":
    main()
