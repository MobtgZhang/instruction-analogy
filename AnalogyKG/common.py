"""
共享配置、Prompt、LLM 客户端、网络搜索与文章加载。
供 extract_triples / validate_triples / correct_triples 使用。
"""
import os
import json

# 配置文件目录：./config（优先脚本所在目录下，其次当前工作目录下）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIRS = (
    os.path.join(_SCRIPT_DIR, "config"),
    os.path.join(os.getcwd(), "config"),
)
CONFIG_NAMES = tuple(os.path.join(d, "api.json") for d in CONFIG_DIRS)

# 内置 prompt 默认值（当 config 下无对应文件时使用）
_DEFAULT_EXTRACTION_PROMPT = """你是一个从文本中抽取「类比三元组」的专家。请根据下面给出的文章内容，尽可能抽取出三类类比信息，并以 JSON 列表形式输出。

## 文章
标题：{title}
语言：{lang}
正文：
{text}

## 抽取要求
请从文章中识别并抽取以下三类类比（若某类在文中无明显体现可返回空列表）：

1. **实体类比（三元组）**：两个实体在属性或角色上形成类比，例如「北京 : 中国 :: 东京 : 日本」。
2. **关系类比（三元组）**：两个关系在语义或结构上形成类比，例如「写 : 作家 :: 画 : 画家」。
3. **子图类比（三元组）**：两个子图/情境在结构或事件上形成类比，例如「A 国首都—A 国」与「B 国首都—B 国」的对应结构。

对每一个类比项，请提供以下 5 个字段（均需填写详细信息，不要省略）：
- **类比A**：类比中的第一个对象/关系/子图（简短名称）。
- **类比B**：类比中的第二个对象/关系/子图（简短名称）。
- **类比A属性(详细信息)**：A 的完整属性、上下文或结构描述。
- **类比B属性(详细信息)**：B 的完整属性、上下文或结构描述。
- **类比AB关系(详细信息)**：A 与 B 之间的类比关系说明，以及为何构成类比。

## 输出格式
请仅输出一个合法的 JSON 对象，不要包含其他说明文字。格式如下：
{{
  "entity_analogy_triples": [
    {{
      "元素A": "...",
      "元素B": "...",
      "元素A属性(详细信息)": "...",
      "元素B属性(详细信息)": "...",
      "类比关系(详细信息)": "..."
    }},
    ...
  ],
  "relation_analogy_triples": [
    {{
      "元素A": "...",
      "元素B": "...",
      "元素A属性(详细信息)": "...",
      "元素B属性(详细信息)": "...",
      "类比关系(详细信息)": "..."
    }},
    ...
  ],
  "subgraph_analogy_triples": [
    {{
      "元素A": "...",
      "元素B": "...",
      "元素A属性(详细信息)": "...",
      "元素B属性(详细信息)": "...",
      "类比关系(详细信息)": "..."
    }},
    ...
  ]
}}

若某类没有合适的类比，对应列表请设为 []。请开始输出 JSON："""

_DEFAULT_VALIDATION_PROMPT = """你是一个类比质量审核专家。请根据「文章摘要」和「网络搜索参考」（如有），判断下面三类类比三元组的**属性描述**和**类比关系**是否合理、正确。

## 文章摘要
标题：{title}
语言：{lang}
正文前 500 字：{text_preview}

## 网络搜索参考（用于核对实体/关系事实）
{search_context}

## 待验证的三类类比（JSON）
{analogy_json}

## 验证要求
对每一类（实体类比、关系类比、子图类比）中的每一项，请判断：
1. **元素A、元素B** 是否在文中有依据或常识正确；
2. **元素A属性、元素B属性** 描述是否准确、完整；
3. **类比关系** 是否成立（A 与 B 是否确实构成合理类比）。

请仅输出一个 JSON 对象，不要其他说明：
{{
  "all_valid": true 或 false,
  "entity_analogy_triples": {{ "valid": true/false, "issues": ["问题1", "问题2"] 或 [] }},
  "relation_analogy_triples": {{ "valid": true/false, "issues": ["问题1"] 或 [] }},
  "subgraph_analogy_triples": {{ "valid": true/false, "issues": ["问题1"] 或 [] }}
}}
若全部正确则 all_valid 为 true 且各 issues 为空。请开始输出 JSON："""

_DEFAULT_CORRECTION_PROMPT = """你是一个类比修正专家。以下是从某篇文章中抽取的类比三元组，但验证发现有问题。请根据「验证反馈」和「网络搜索参考」修正 JSON，保持原有输出格式，只修正错误处。

## 文章摘要
标题：{title}
正文前 500 字：{text_preview}

## 网络搜索参考（用于修正时的事实核对）
{search_context}

## 当前类比 JSON（有误）
{analogy_json}

## 验证反馈
{feedback}

## 要求
输出修正后的完整 JSON 对象，格式与输入相同（entity_analogy_triples, relation_analogy_triples, subgraph_analogy_triples），仅输出 JSON，不要其他文字："""

_PROMPTS_CACHE = None

WIKIPEDIA_EN_JSONL = "wikipedia_en_20000.jsonl"
WIKIPEDIA_ZH_JSONL = "wikipedia_zh_20000.jsonl"


def load_config() -> dict:
    """从 ./config/api.json 加载配置（API Key、路径等）。不存在或解析失败时返回空 dict。"""
    for path in CONFIG_NAMES:
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  警告：读取配置失败 {path}: {e}")
            break
    return {}


def _load_prompt_file(filename: str) -> str | None:
    """从 ./config 目录读取 prompt 文件内容，未找到返回 None。"""
    for config_dir in CONFIG_DIRS:
        path = os.path.join(config_dir, filename)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except OSError as e:
                print(f"  警告：读取 prompt 失败 {path}: {e}")
    return None


def get_prompts() -> dict:
    """从 ./config 下的 extraction_prompt.txt、validation_prompt.txt、correction_prompt.txt 加载；
    若文件不存在则使用内置默认。返回 {"extraction", "validation", "correction"}。"""
    global _PROMPTS_CACHE
    if _PROMPTS_CACHE is not None:
        return _PROMPTS_CACHE
    extraction = _load_prompt_file("extraction_prompt.txt") or _DEFAULT_EXTRACTION_PROMPT
    validation = _load_prompt_file("validation_prompt.txt") or _DEFAULT_VALIDATION_PROMPT
    correction = _load_prompt_file("correction_prompt.txt") or _DEFAULT_CORRECTION_PROMPT
    _PROMPTS_CACHE = {"extraction": extraction, "validation": validation, "correction": correction}
    return _PROMPTS_CACHE


def web_search_optional(query: str, max_results: int = 3) -> str:
    """可选网络搜索，返回拼接的摘要文本；未安装搜索库或失败时返回空字符串。"""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return ""
        return "\n".join(
            f"[{i+1}] {r.get('title', '')} | {r.get('body', '')[:200]}"
            for i, r in enumerate(results)
        )
    except Exception:
        return ""


def get_llm_generator(model_name: str, api_key: str = None, base_url: str = None):
    """
    使用 OpenAI 兼容 API 请求 LLM，返回一个函数 generator(prompt) -> str。
    api_key: 默认从环境变量 OPENAI_API_KEY 读取。
    base_url: 可选，用于兼容 OpenAI 格式的第三方接口。
    若加载失败则返回 None，调用方可用占位或跳过。
    """
    try:
        from openai import OpenAI

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            print("  未设置 OPENAI_API_KEY，无法使用 OpenAI LLM。")
            return None
        client = OpenAI(api_key=key, base_url=base_url)

        def generate(prompt: str, max_tokens: int = 4096) -> str:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return (response.choices[0].message.content or "").strip()

        return generate
    except Exception as e:
        print(f"  OpenAI LLM 初始化失败 ({model_name}): {e}")
        return None


def parse_json_from_llm(raw: str) -> dict:
    """从 LLM 回复中截取并解析 JSON。"""
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        raw = raw[start:end]
    return json.loads(raw)


def load_wikipedia_articles(data_path: str):
    """
    从 data_path 目录下读取 wikipedia_en_20000.jsonl 和 wikipedia_zh_20000.jsonl，
    逐条 yield 文章。每条为 dict：id, url, title, text, lang。
    """
    for filename, lang in [(WIKIPEDIA_EN_JSONL, "en"), (WIKIPEDIA_ZH_JSONL, "zh")]:
        filepath = os.path.join(data_path, filename)
        if not os.path.exists(filepath):
            print(f"  跳过不存在的文件: {filepath}")
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    article = {
                        "id": obj.get("id", ""),
                        "url": obj.get("url", ""),
                        "title": obj.get("title", ""),
                        "text": obj.get("text", ""),
                        "lang": obj.get("lang", lang),
                    }
                    yield article
                except json.JSONDecodeError as e:
                    print(f"  解析失败 {filename}: {e}")
