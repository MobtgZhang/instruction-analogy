"""
类比三元组构建入口：语料准备 + 抽取 → 验证(web_search) → 修正(web_search)与写入。
"""
import os
import json
import argparse
from datasets import load_dataset  # noqa: E402

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HUGGINGFACE_HUB_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from common import load_config, load_wikipedia_articles, get_llm_generator
from extract_triples import extract_analogy_triples_from_article
from correct_triples import validate_and_correct_until_done


def sample_dataset(ds, n: int, seed: int = 42):
    """在给定数据集中按样本数 n 进行随机下采样。"""
    n = min(n, len(ds))
    return ds.shuffle(seed=seed).select(range(n))


def _load_with_retry(name, config, split, cache_dir, max_retries=3):
    """尝试在线加载数据集，失败后重试；全部失败则回退到本地缓存。"""
    import time
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return load_dataset(
                name, config, split=split, cache_dir=cache_dir, trust_remote_code=True,
            )
        except (ConnectionError, OSError) as exc:
            last_exc = exc
            wait = 2 ** attempt
            print(f"  ⚠ 第 {attempt}/{max_retries} 次尝试失败：{exc}")
            if attempt < max_retries:
                print(f"    {wait}s 后重试 …")
                time.sleep(wait)
    print("  → 在线加载全部失败，尝试使用本地缓存 …")
    try:
        from datasets import DownloadMode
        return load_dataset(
            name, config, split=split, cache_dir=cache_dir,
            download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS,
            trust_remote_code=True,
        )
    except Exception:
        pass
    raise RuntimeError(
        f"无法加载数据集 {name}/{config}（在线和本地缓存均失败）。\n"
        f"最后一次错误：{last_exc}"
    )


def build_balanced_corpus(
    data_path: str,
    dataset_name: str,
    cache_dir: str,
    out_dir: str,
    sample_per_lang: int,
):
    """构建中英文类别均衡的 Wikipedia 文本语料，保存为 JSONL。"""
    load_zh_file = os.path.join(data_path, "wikipedia_zh_20000.jsonl")
    if not os.path.exists(load_zh_file):
        print("[1/5] 正在加载中文数据集 …")
        zh_dataset = _load_with_retry(dataset_name, "20231101.zh", "train", cache_dir)
        print(f"[3/5] 中文原始样本规模：{len(zh_dataset):,d}")
        zh_sampled = sample_dataset(zh_dataset, sample_per_lang, seed=42)
        zh_sampled = zh_sampled.map(
            lambda _: {"lang": "zh"},
            desc="为中文样本新增语言标记列 lang",
        )
        output_path_zh = os.path.join(out_dir, f"wikipedia_zh_{len(zh_sampled)}.jsonl")
        zh_sampled.to_json(output_path_zh, lines=True, force_ascii=False)
        print(f"[5/5] 中文子集已写入：{output_path_zh}")
    else:
        print("[1/5] 中文数据集已存在，跳过加载")

    load_en_file = os.path.join(data_path, "wikipedia_en_20000.jsonl")
    if not os.path.exists(load_en_file):
        print("[2/5] 正在加载英文数据集 …")
        en_dataset = _load_with_retry(dataset_name, "20231101.en", "train", cache_dir)
        print(f"[3/5] 英文原始样本规模：{len(en_dataset):,d}")
        en_sampled = sample_dataset(en_dataset, sample_per_lang, seed=43)
        en_sampled = en_sampled.map(
            lambda _: {"lang": "en"},
            desc="为英文样本新增语言标记列 lang",
        )
        output_path_en = os.path.join(out_dir, f"wikipedia_en_{len(en_sampled)}.jsonl")
        en_sampled.to_json(output_path_en, lines=True, force_ascii=False)
        print(f"[5/5] 英文子集已写入：{output_path_en}")
    else:
        print("[2/5] 英文数据集已存在，跳过加载")


def parse_args():
    cfg = load_config()

    def _get(key: str, default):
        val = cfg.get(key)
        if val is None:
            return default
        return val

    parser = argparse.ArgumentParser(
        description="构建/读取平衡的中英文 Wikipedia 语料，并抽取·验证·修正类比三元组",
    )
    parser.add_argument("--force-rebuild", action="store_true", help="强制重新构建语料")
    parser.add_argument("--data-path", default=_get("data_path", "data/balanced"), help="数据集目录")
    parser.add_argument("--cache-dir", default=_get("cache_dir", "data/wikipedia"), help="缓存目录")
    parser.add_argument("--out-dir", default=_get("out_dir", "data/balanced"), help="输出目录")
    parser.add_argument("--llm-model", default=_get("llm_model", "gpt-4o-mini"), help="OpenAI 模型名")
    parser.add_argument("--dataset-name", default=_get("dataset_name", "wikimedia/wikipedia"), help="数据集名称")
    parser.add_argument("--sample-per-lang", type=int, default=_get("sample_per_lang", 20000), help="每语言样本数")
    parser.add_argument("--max-articles", type=int, default=_get("max_articles", 0), help="最多处理文章数，0 表示不限制")
    parser.add_argument("--openai-api-key", default=_get("openai_api_key", ""), help="OpenAI API Key")
    parser.add_argument("--openai-base-url", default=_get("openai_base_url", ""), help="OpenAI 兼容 API base_url")
    parser.add_argument("--validation-max-retries", type=int, default=_get("validation_max_retries", 3), help="验证不通过时最多修正重试次数")
    parser.add_argument("--no-web-search", action="store_true", default=_get("no_web_search", False), help="验证/修正阶段不使用网络搜索")
    return parser.parse_args()


def build_analogy_triples(
    data_path: str,
    llm_model: str,
    out_dir: str,
    max_articles: int = 0,
    validation_max_retries: int = 3,
    use_web_search: bool = True,
    openai_api_key: str = None,
    openai_base_url: str = None,
):
    """
    从 data_path 下的 wikipedia_*_20000.jsonl 读取文章，
    1) LLM 抽取三类类比
    2) 验证（web_search + LLM）
    3) 通过则写入；不通过则用 web_search 修正后再验证，直至通过或达最大重试
    """
    os.makedirs(out_dir, exist_ok=True)
    entity_out = os.path.join(out_dir, "entity_analogy_triples.jsonl")
    relation_out = os.path.join(out_dir, "relation_analogy_triples.jsonl")
    subgraph_out = os.path.join(out_dir, "subgraph_analogy_triples.jsonl")
    for p in (entity_out, relation_out, subgraph_out):
        if os.path.exists(p):
            open(p, "w", encoding="utf-8").close()

    llm_generate = get_llm_generator(
        llm_model,
        api_key=openai_api_key or "",
        base_url=openai_base_url or None,
    )
    if llm_generate is None:
        print("  未可用 LLM，跳过类比抽取。请设置 OPENAI_API_KEY 或 --openai-api-key。")
        return

    count = 0
    for article in load_wikipedia_articles(data_path):
        if max_articles and count >= max_articles:
            break
        count += 1
        title_short = (article.get("title", "") or "")[:50]
        print(f"  处理第 {count} 篇: {title_short}... (lang={article.get('lang')})")

        result = extract_analogy_triples_from_article(article, llm_generate)
        result, valid, corrections = validate_and_correct_until_done(
            article,
            result,
            llm_generate,
            validation_max_retries=validation_max_retries,
            use_web_search=use_web_search,
        )
        if valid:
            if corrections > 0:
                print(f"    第 {corrections} 次修正后验证通过。")
        else:
            print("    已达最大重试次数，保留当前结果写入。")

        def add_meta(rec, art):
            rec["_article_id"] = art.get("id", "")
            rec["_article_title"] = art.get("title", "")
            rec["_article_lang"] = art.get("lang", "")
            return rec

        for rec in result.get("entity_analogy_triples") or []:
            with open(entity_out, "a", encoding="utf-8") as f:
                f.write(json.dumps(add_meta(rec, article), ensure_ascii=False) + "\n")
        for rec in result.get("relation_analogy_triples") or []:
            with open(relation_out, "a", encoding="utf-8") as f:
                f.write(json.dumps(add_meta(rec, article), ensure_ascii=False) + "\n")
        for rec in result.get("subgraph_analogy_triples") or []:
            with open(subgraph_out, "a", encoding="utf-8") as f:
                f.write(json.dumps(add_meta(rec, article), ensure_ascii=False) + "\n")

    print(f"类比三元组抽取完成，共处理 {count} 篇。输出：")
    print(f"  - {entity_out}")
    print(f"  - {relation_out}")
    print(f"  - {subgraph_out}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    build_balanced_corpus(
        args.data_path,
        args.dataset_name,
        args.cache_dir,
        args.out_dir,
        args.sample_per_lang,
    )
    build_analogy_triples(
        args.data_path,
        args.llm_model,
        args.out_dir,
        max_articles=args.max_articles,
        validation_max_retries=args.validation_max_retries,
        use_web_search=not args.no_web_search,
        openai_api_key=args.openai_api_key or None,
        openai_base_url=args.openai_base_url or None,
    )


if __name__ == "__main__":
    main()
