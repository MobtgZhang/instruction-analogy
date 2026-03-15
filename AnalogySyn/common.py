"""
共享：配置加载、Prompt 加载、LLM 客户端、类比数据读取、百科检索、JSON 解析。
"""
import os
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = ROOT_DIR / "config"
CONFIG_NAMES = [CONFIG_DIR / "api.json", Path(os.getcwd()) / "config" / "api.json"]

_PROMPTS_CACHE: dict | None = None


def load_config() -> dict:
    """从 config/api.json 加载配置。不存在或解析失败时返回空 dict。"""
    for path in CONFIG_NAMES:
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  警告：读取配置失败 {path}: {e}")
            break
    return {}


def get_prompts() -> dict:
    """从 config 目录加载 synthesis/validate/correction/classify 四个 prompt。"""
    global _PROMPTS_CACHE
    if _PROMPTS_CACHE is not None:
        return _PROMPTS_CACHE
    out = {}
    for name in ("synthesis_prompt", "validate_prompt", "correction_prompt", "classify_prompt"):
        path = CONFIG_DIR / f"{name}.txt"
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                out[name] = f.read()
        else:
            out[name] = ""
    _PROMPTS_CACHE = out
    return out


def get_llm_generator(
    model_name: str = None,
    api_key: str = None,
    base_url: str = None,
):
    """返回 generate(prompt, max_tokens) -> str，失败返回 None。"""
    try:
        from openai import OpenAI
        cfg = load_config()
        key = api_key or cfg.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
        if not key:
            print("  未设置 openai_api_key / OPENAI_API_KEY。")
            return None
        base = base_url or cfg.get("openai_base_url") or None
        model = model_name or cfg.get("llm_model", "gpt-4o-mini")
        client = OpenAI(api_key=key, base_url=base)

        def generate(prompt: str, max_tokens: int = 4096) -> str:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return (r.choices[0].message.content or "").strip()

        return generate
    except Exception as e:
        print(f"  LLM 初始化失败: {e}")
        return None


def _norm_analogy_record(rec: dict) -> dict:
    """统一不同 key 名称（元素A/类比A 等）为固定字段。"""
    a = rec.get("元素A") or rec.get("类比A") or ""
    b = rec.get("元素B") or rec.get("类比B") or ""
    attr_a = rec.get("元素A属性(详细信息)") or rec.get("类比A属性(详细信息)") or ""
    attr_b = rec.get("元素B属性(详细信息)") or rec.get("类比B属性(详细信息)") or ""
    rel = rec.get("类比关系(详细信息)") or rec.get("类比AB关系(详细信息)") or ""
    return {
        "analogy_a": a,
        "analogy_b": b,
        "attr_a": attr_a,
        "attr_b": attr_b,
        "relation": rel,
        "source": rec.get("_article_title", ""),
        "article_id": rec.get("_article_id", ""),
        "lang": rec.get("_article_lang", ""),
    }


def load_analogy_data(data_dir: str) -> list[dict]:
    """
    从目录中读取类比数据。目录下可有：
    - entity_analogy_triples.jsonl
    - relation_analogy_triples.jsonl
    - subgraph_analogy_triples.jsonl
    或单个 all_analogy.jsonl。
    返回列表，每项为归一化后的 dict，并带 order 与 category。
    """
    data_dir = Path(data_dir)
    out = []
    files = [
        ("entity_analogy_triples.jsonl", "1st-order", "entity"),
        ("relation_analogy_triples.jsonl", "2nd-order", "relation"),
        ("subgraph_analogy_triples.jsonl", "3rd-order", "subgraph"),
    ]
    single = data_dir / "all_analogy.jsonl"
    if single.exists():
        with open(single, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    norm = _norm_analogy_record(rec)
                    norm["order"] = rec.get("order", "1st-order")
                    norm["category"] = rec.get("category", "entity")
                    out.append(norm)
                except json.JSONDecodeError:
                    continue
        return out

    for filename, order, category in files:
        path = data_dir / filename
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    norm = _norm_analogy_record(rec)
                    norm["order"] = order
                    norm["category"] = category
                    out.append(norm)
                except json.JSONDecodeError:
                    continue
    return out


def fetch_encyclopedia_info(analogy_a: str, analogy_b: str, max_results: int = 2) -> str:
    """根据类比两端实体/概念获取在线百科或检索摘要。失败返回空字符串。"""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            parts = []
            for q in [f"{analogy_a} 百科", f"{analogy_b} 百科"]:
                results = list(ddgs.text(q, max_results=max_results))
                if results:
                    parts.append(q + "：\n" + "\n".join(
                        r.get("body", "")[:300] for r in results[:2]
                    ))
            return "\n\n".join(parts) if parts else ""
    except Exception:
        return ""


def parse_json_from_llm(raw: str) -> dict:
    """从 LLM 回复中截取并解析 JSON。"""
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        raw = raw[start:end]
    return json.loads(raw)
