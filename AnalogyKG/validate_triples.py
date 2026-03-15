"""
2. 验证三元组（需 web_search）
根据文章摘要与网络搜索参考，用 LLM 判断三类类比三元组的属性与类比关系是否正确。
"""
import json

from common import get_prompts, parse_json_from_llm, web_search_optional


def gather_search_context(article: dict, result: dict) -> str:
    """根据文章和抽取结果构造搜索查询并做网络搜索，返回拼接的参考文本。"""
    title = article.get("title", "")
    lang = article.get("lang", "zh")
    parts = []
    if title:
        parts.append(web_search_optional(f"{title} 百科 简介", max_results=2))
    for key in ("entity_analogy_triples", "relation_analogy_triples", "subgraph_analogy_triples"):
        for item in (result.get(key) or [])[:2]:
            a, b = item.get("元素A", ""), item.get("元素B", "")
            if a or b:
                q = f"{a} {b} 关系 类比" if lang == "zh" else f"{a} {b} relationship analogy"
                parts.append(web_search_optional(q, max_results=1))
    combined = "\n\n".join(p for p in parts if p)
    return combined if combined else "（未进行网络搜索或搜索无结果）"


def validate_analogy_triples(
    article: dict, result: dict, llm_generate, use_web_search: bool = True
) -> tuple[bool, dict]:
    """
    验证三类类比三元组的属性与类比关系是否正确。
    使用网络搜索（必须用于验证）+ LLM 释义判断。
    返回 (是否全部通过, 验证结果 JSON，含 all_valid 与各类型的 issues)。
    """
    text_preview = (article.get("text", "") or "")[:500]
    search_context = gather_search_context(article, result) if use_web_search else "（未使用网络搜索）"
    analogy_json_str = json.dumps(result, ensure_ascii=False, indent=2)

    prompt = get_prompts()["validation"].format(
        title=article.get("title", ""),
        lang=article.get("lang", ""),
        text_preview=text_preview,
        search_context=search_context,
        analogy_json=analogy_json_str,
    )
    raw = llm_generate(prompt, max_tokens=1024) if llm_generate else "{}"
    try:
        verdict = parse_json_from_llm(raw)
        all_valid = verdict.get("all_valid", False)
        return all_valid, verdict
    except json.JSONDecodeError:
        return False, {"all_valid": False, "issues": ["验证输出解析失败"]}
