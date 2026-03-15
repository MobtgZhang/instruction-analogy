"""
3. 修正与写入三元组
若验证通过则写入结果；若不通过则基于 web_search 上下文重新修正，并再次验证。
"""
import json

from common import get_prompts, parse_json_from_llm
from validate_triples import validate_analogy_triples, gather_search_context


def correct_analogy_triples(
    article: dict,
    result: dict,
    feedback: dict,
    llm_generate,
    search_context: str = None,
) -> dict:
    """
    根据验证反馈与网络搜索参考用 LLM 修正类比三元组，返回修正后的 result。
    若未传入 search_context，则通过 gather_search_context(article, result) 做 web_search 获取。
    """
    if search_context is None:
        search_context = gather_search_context(article, result)
    text_preview = (article.get("text", "") or "")[:500]
    analogy_json_str = json.dumps(result, ensure_ascii=False, indent=2)
    feedback_str = json.dumps(feedback, ensure_ascii=False, indent=2)

    prompt_template = get_prompts()["correction"]
    kwargs = {
        "title": article.get("title", ""),
        "text_preview": text_preview,
        "analogy_json": analogy_json_str,
        "feedback": feedback_str,
    }
    if "{search_context}" in prompt_template:
        kwargs["search_context"] = search_context or ""
    prompt = prompt_template.format(**kwargs)
    raw = llm_generate(prompt, max_tokens=2048) if llm_generate else "{}"
    try:
        corrected = parse_json_from_llm(raw)
        for key in ("entity_analogy_triples", "relation_analogy_triples", "subgraph_analogy_triples"):
            if key not in corrected or not isinstance(corrected[key], list):
                corrected[key] = result.get(key, [])
        return corrected
    except json.JSONDecodeError:
        return result


def validate_and_correct_until_done(
    article: dict,
    result: dict,
    llm_generate,
    validation_max_retries: int = 3,
    use_web_search: bool = True,
) -> tuple[dict, bool, int]:
    """
    验证三元组：通过则返回 (result, True, 修正次数)；
    不通过则用 web_search 修正后再次验证，最多重试 validation_max_retries 次，
    最后返回 (最终 result, 是否通过验证, 修正次数)。
    """
    current = result
    corrections = 0
    for attempt in range(validation_max_retries + 1):
        valid, feedback = validate_analogy_triples(
            article, current, llm_generate, use_web_search=use_web_search
        )
        if valid:
            return current, True, corrections
        if attempt < validation_max_retries:
            print(f"    验证未通过（第 {attempt + 1} 次），使用 web_search 修正...")
            search_context = gather_search_context(article, current) if use_web_search else ""
            current = correct_analogy_triples(
                article, current, feedback, llm_generate, search_context=search_context
            )
            corrections += 1
    return current, False, corrections
