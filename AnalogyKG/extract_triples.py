"""
1. LLM 抽取三元组
从单篇文章中用 LLM 抽取实体类比、关系类比、子图类比三类三元组。
"""
import json

from common import get_prompts, get_llm_generator, parse_json_from_llm


def extract_analogy_triples_from_article(article: dict, llm_generate) -> dict:
    """对单篇文章用 LLM 抽取三类类比三元组，返回包含 entity/relation/subgraph 的 dict。"""
    prompt = get_prompts()["extraction"].format(
        title=article.get("title", ""),
        lang=article.get("lang", ""),
        text=(article.get("text", "") or "")[:8000],
    )
    raw = llm_generate(prompt, max_tokens=2048) if llm_generate else "{}"
    try:
        return parse_json_from_llm(raw)
    except json.JSONDecodeError:
        return {
            "entity_analogy_triples": [],
            "relation_analogy_triples": [],
            "subgraph_analogy_triples": [],
        }
