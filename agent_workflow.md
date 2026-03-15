# Analogy Dataset Synthesis Agent Workflow

## Overview

This workflow automates the synthesis of high-quality analogy-reasoning datasets from documents. Each sample has the form `(query, think, answer)`. Datasets are organized **longitudinally** by analogy order and **transversely** by Bloom’s cognitive taxonomy.

---

## Analogy Dimensions

### Longitudinal: Analogy Order

| Order | Name | Description | Triple structure |
|-------|------|-------------|------------------|
| 1st | Entity analogy | Direct similarity or correspondence between two entities | `(entity_A, relation, entity_B)` |
| 2nd | Relational analogy | Structural similarity between two entity relations | `(r1(A,B), ≅, r2(C,D))` |
| 3rd | Subgraph analogy | Structural isomorphism or similarity between two sub-knowledge-graphs | `SubGraph_1 ≅ SubGraph_2` |

### Transverse: Bloom Taxonomy

| Cognitive level | Analogy type | Description |
|-----------------|--------------|-------------|
| **Factual** | Entity analogy | Compare similarity or correspondence between concrete entities |
| | Attribute analogy | Compare similarity of entity attributes or features |
| | Temporal analogy | Compare temporal order or duration of events or processes |
| | Spatial analogy | Compare location, structure, or spatial relations |
| | Quantitative analogy | Compare quantity, measure, or proportion |
| | Event analogy | Compare occurrence, cause-effect, or sequence of events |
| **Conceptual** | Relational analogy | Analogize relation patterns between concepts |
| | System analogy | Compare similarity of complex systems or structures |
| | Functional analogy | Compare functional roles of different objects or concepts |
| | Causal analogy | Analogize cause-effect patterns |
| | Proportional analogy | Compare proportional or regular relations in quantity or nature |
| | Metaphorical analogy | Compare figurative or symbolic similarity of abstract concepts |

---

## Overall Pipeline

```
HuggingFace document dataset
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│         Sub-Module 0: KG Builder                          │
│                                                          │
│  Step-A Dataset download & preprocessing                 │
│      │                                                   │
│      ▼                                                   │
│  Step-B Analogy-oriented triple extraction (6 relation types) │
│      │                                                   │
│      ▼                                                   │
│  Step-C Web Search verification → correction / confidence │
│      │                                                   │
│      ▼                                                   │
│  Knowledge graph G (high-quality, analogy-ready)          │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│        Phase 1: Analogy pattern mining  │
│  1st/2nd/3rd-order candidate discovery   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│   Sub-Module 2: Data synthesis (Agent A) │
│  (query, think, answer) generation      │
│  Explicit + implicit analogy modes       │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│   Sub-Module 3: Quality validation (Agent B) │
│  Validation → correction → re-generation │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│   Sub-Module 4: Bloom classification (Agent C) │
│  Auto classify → verify → annotate       │
└─────────────────────────────────────────┘
    │
    ▼
Final dataset (Bloom-annotated)
```

---

## Sub-Module 0: Knowledge Graph Builder (KG Builder)

> This sub-module is the data foundation of the pipeline and is packaged as a reusable workflow. Input: public document datasets on HuggingFace. Output: structured knowledge graph `G` for analogy reasoning.

### KG Builder flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                   KG Builder Sub-Workflow                            │
│                                                                      │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────────┐   │
│  │  Step-A     │    │  Step-B      │    │  Step-C               │   │
│  │  Data       │───►│  Triple      │───►│  Web verification     │   │
│  │  collection │    │  extraction │    │  Agent                 │   │
│  └─────────────┘    └──────────────┘    └──────────┬────────────┘   │
│                            ▲                       │                 │
│                            │      ┌────────────────┘                │
│                            │      │  Verification result             │
│                            │      ▼                                  │
│                            │  ┌────────────────────┐                │
│                            │  │  PASS → into graph │                │
│                            │  │  FAIL → correct & re-verify         │
│                            │  │  LOW_CONF → tag    │                │
│                            │  └────────────────────┘                │
│                            │           │                             │
│                            └───────────┘ (feedback to Step-B)        │
│                                                                      │
│  Output: Knowledge graph G = (V, E, L) + triple confidence table    │
└──────────────────────────────────────────────────────────────────────┘
```

---

### Step-A: Document dataset collection Agent

#### A.1 Recommended HuggingFace datasets

The following datasets are selected for multi-domain coverage and rich entity relations, suitable for building analogy-oriented knowledge graphs:

| 优先级 | 数据集名称 | HuggingFace 路径 | 领域 | 推荐理由 |
|--------|-----------|-----------------|------|----------|
| ⭐⭐⭐ | **Wikipedia（英/中）** | `wikimedia/wikipedia`（`20231101.en` / `20231101.zh`） | 综合百科 | 覆盖面最广，实体关系密集，跨域类比素材丰富 |
| ⭐⭐⭐ | **WikiData Triples** | `Babelscape/rebel-dataset` | 结构化事实 | 已含 220+ 关系类型的三元组，可直接复用 |
| ⭐⭐⭐ | **Scientific Papers** | `allenai/s2orc`（子集） | 科学 | 科学类比（物理/生物/化学）素材密集 |
| ⭐⭐ | **SimpleWiki** | `wikipedia` (`20231101.simple`) | 简化百科 | 语言简单，实体关系表达清晰，易于三元组抽取 |
| ⭐⭐ | **Open Book QA** | `allenai/openbookqa` | 常识科学 | 包含大量科学概念的类比关系描述 |
| ⭐⭐ | **ConceptNet5** | `conceptnet5/conceptnet5` | 常识知识图谱 | 原生三元组格式，关系类型语义明确（IsA, UsedFor, CapableOf 等） |
| ⭐⭐ | **Wikibooks** | `wikimedia/wikibooks`（`en`） | 教科书 | 学科知识系统性强，适合构建领域内类比 |
| ⭐ | **TED Talks** | `ted_talks_iwslt` | 科普演讲 | 大量隐喻和类比表达，适合隐式类比挖掘 |

**推荐组合策略**：
- **核心知识源**：`wikimedia/wikipedia (en/zh)` + `Babelscape/rebel-dataset`
- **科学类比补充**：`allenai/s2orc`（选取 CS、Physics、Biology 领域子集）
- **常识类比补充**：`conceptnet5/conceptnet5`（可直接作为图谱种子）
- **隐喻类比补充**：`wikimedia/wikibooks (en)` 中的理科教材章节

#### A.2 数据集下载与抽样策略

```
数据集下载 Agent 执行流程：

1. 使用 datasets 库流式加载，避免全量下载
   └─ from datasets import load_dataset
      ds = load_dataset("wikimedia/wikipedia", "20231101.en", streaming=True)

2. 领域均衡抽样（避免单一领域主导）
   目标领域权重分配：
   ├─ 自然科学（物理/化学/生物）：30%
   ├─ 社会科学（历史/经济/政治）：25%
   ├─ 技术工程（计算机/机械/建筑）：20%
   ├─ 文化艺术（文学/音乐/哲学）：15%
   └─ 日常生活（地理/食物/运动）：10%

3. 文档过滤条件
   ├─ 文档长度：500 ~ 5000 tokens（过滤过短/过长文档）
   ├─ 实体密度：段落中命名实体数量 ≥ 3
   └─ 语言质量：困惑度 ≤ 阈值（排除低质量页面）

4. 输出：结构化文档集合 D_raw
   格式：{"doc_id", "title", "domain", "text", "source"}
```

#### A.3 文档预处理

```
预处理流程：

├─ [1] 格式清洗：去除 wiki 标记语法、HTML 标签、引用符号
├─ [2] 语义分块（Semantic Chunking）
│       方法：基于句子嵌入相似度切分（相邻句相似度 < 0.5 时切分）
│       块大小：200 ~ 512 tokens，保留 50 token 重叠窗口
├─ [3] 实体识别（NER）
│       模型推荐：spaCy en_core_web_trf 或 GLiNER（通用 NER）
│       识别类型：PERSON, ORG, GPE, EVENT, CONCEPT, QUANTITY, DATE
├─ [4] 共指消解
│       工具：FastCoref 或 LingMess
│       将"它"、"该机构"等代词映射到具体实体
└─ [5] 输出
        结构化块列表：[{"chunk_id", "doc_id", "text", "entities": [...]}]
```

---

### Step-B：类比导向三元组抽取 Agent

> 本步骤是知识图谱构建的核心。相比通用三元组抽取，**类比导向抽取**额外关注：
> 跨域可类比性、关系的结构对称性、属性的可比较性。

#### B.1 六大类比关系类型定义

| 类型编号 | 关系类型 | 谓词示例 | 类比价值 | 抽取优先级 |
|----------|----------|----------|----------|------------|
| T1 | **功能关系** | `performs_function`, `used_for`, `enables` | 功能类比的直接来源 | ⭐⭐⭐ |
| T2 | **结构/组成关系** | `part_of`, `composed_of`, `contains` | 系统类比、子图类比 | ⭐⭐⭐ |
| T3 | **因果关系** | `causes`, `leads_to`, `results_in`, `inhibits` | 因果类比 | ⭐⭐⭐ |
| T4 | **属性关系** | `has_property`, `has_size`, `has_value` | 属性类比、量化类比 | ⭐⭐ |
| T5 | **层级/分类关系** | `is_a`, `subtype_of`, `instance_of` | 实体类比、关系类比 | ⭐⭐ |
| T6 | **时序/空间关系** | `precedes`, `follows`, `located_in`, `adjacent_to` | 时间/空间类比 | ⭐⭐ |

#### B.2 抽取方法（三层流水线）

```
输入：结构化文本块 chunk

Layer 1：规则增强抽取（高精度，低召回）
├─ 使用依存句法分析（spaCy/Stanza）抽取 SVO 结构
├─ 基于触发词模式匹配六大关系类型
│   示例规则：
│   ├─ "X is used for Y"    → (X, used_for, Y)           [T1]
│   ├─ "X consists of Y"    → (X, composed_of, Y)        [T2]
│   ├─ "X causes Y"         → (X, causes, Y)             [T3]
│   ├─ "X is a type of Y"   → (X, subtype_of, Y)         [T5]
│   └─ "X occurs before Y"  → (X, precedes, Y)           [T6]
└─ 输出：高置信度三元组集合 T_rule

Layer 2：LLM 抽取（高召回，中精度）
├─ 提示词框架（Prompt Engineering）：
│   ┌─────────────────────────────────────────────────────────┐
│   │ System: 你是知识图谱构建专家，专注于抽取适合类比推理的   │
│   │         结构化三元组。                                   │
│   │                                                         │
│   │ Task: 从以下文本中抽取三元组，优先关注以下关系类型：      │
│   │   - 功能关系（某物用于/执行什么）                        │
│   │   - 结构/组成关系（某物由什么组成）                      │
│   │   - 因果关系（某事导致/引发什么）                        │
│   │   - 属性关系（某物具有什么特征/性质）                    │
│   │   - 层级关系（某物是什么的子类型）                       │
│   │   - 时序/空间关系（某事发生在何时/何地之前/之后）        │
│   │                                                         │
│   │ 输出格式：JSON 列表                                      │
│   │ [{"subject": "", "predicate": "", "object": "",         │
│   │   "relation_type": "T1~T6", "source_sentence": "",     │
│   │   "confidence": 0.0~1.0}]                              │
│   │                                                         │
│   │ 文本：{chunk_text}                                      │
│   └─────────────────────────────────────────────────────────┘
└─ 输出：LLM 三元组集合 T_llm

Layer 3：融合与去重
├─ 合并 T_rule ∪ T_llm
├─ 语义去重（谓词归一化 + 实体对齐）
├─ 置信度融合：
│   conf_final = w_rule × conf_rule + w_llm × conf_llm
│   （w_rule = 0.6，w_llm = 0.4，若只有一方则直接使用）
└─ 过滤阈值：conf_final < 0.5 的三元组进入待验证队列
```

#### B.3 类比友好性评分

对每条三元组额外计算**类比友好性分数（Analogy-Friendliness Score, AFS）**：

```
AFS = α × cross_domain_score       # 跨域潜力（实体来自不同领域权重高）
    + β × relation_symmetry_score   # 关系对称性（关系是否普遍存在于多个域）
    + γ × attribute_richness_score  # 属性丰富度（实体在图中的属性数量）

其中 α=0.4, β=0.4, γ=0.2

AFS ≥ 0.7 → 高优先级类比候选三元组（优先进入后续类比挖掘）
AFS ∈ [0.4, 0.7) → 普通三元组
AFS < 0.4 → 低类比价值（可保留但后续权重低）
```

**输出**：带类型、置信度、AFS 标注的三元组集合
```json
{
  "triple_id": "t_001",
  "subject": "心脏",
  "predicate": "performs_function",
  "object": "泵送血液",
  "relation_type": "T1",
  "confidence": 0.92,
  "afs": 0.85,
  "source_doc": "doc_042",
  "source_sentence": "心脏的主要功能是泵送血液...",
  "verification_status": "pending"
}
```

---

### Step-C：三元组验证 Agent（Web Search 验证）

> 使用 `web_search` 工具对三元组的事实准确性进行在线验证，
> 以维基百科、百度百科、专业学术页面等权威来源为基准。

#### C.1 web_search 工具定义

```python
def web_search(query: str, sources: list[str] = None, top_k: int = 3) -> dict:
    """
    在线搜索工具，用于验证三元组的事实准确性。

    参数：
        query   : 搜索查询字符串
        sources : 优先搜索的来源域名列表（如 ["wikipedia.org", "baike.baidu.com"]）
        top_k   : 返回结果数量

    返回：
        {
            "query": str,
            "results": [
                {
                    "url": str,
                    "title": str,
                    "snippet": str,       # 摘要文本（~200字）
                    "source_domain": str,
                    "relevance_score": float
                }
            ],
            "answer_box": str | None,     # 直接答案（若搜索引擎提供）
            "search_timestamp": str
        }

    工具实现建议：
        - 生产环境：接入 Bing Search API / Google Custom Search API / Tavily API
        - 开发环境：使用 DuckDuckGo Search（免费，无需 API Key）
        - 备用方案：SerpAPI（支持多搜索引擎）
    """
```

#### C.2 验证查询构造策略

针对不同关系类型，构造不同的验证查询：

| 关系类型 | 查询模板 | 验证来源优先级 |
|----------|----------|----------------|
| T1 功能关系 | `"{subject} 的功能 OR used for"` | Wikipedia > 专业教材网站 |
| T2 结构关系 | `"{subject} 组成 OR composed of OR structure"` | Wikipedia > Encyclopedia Britannica |
| T3 因果关系 | `"{subject} 导致 {object} OR {subject} causes {object}"` | 学术论文 > Wikipedia |
| T4 属性关系 | `"{subject} {predicate} value"` | Wikipedia Infobox > 专业数据库 |
| T5 层级关系 | `"{subject} is a type of OR 是一种"` | Wikipedia > Wikidata |
| T6 时序关系 | `"{subject} 时间 OR {subject} year OR date"` | Wikipedia > 历史数据库 |

#### C.3 验证执行流程

```
对每条三元组 t = (s, p, o, conf, afs)：
    │
    ├─→ [优先级过滤]
    │     若 conf ≥ 0.85 且 afs < 0.5 → 跳过验证，直接标注 LOW_PRIORITY
    │     若 conf ≥ 0.90 → 快速验证（仅 1 次搜索）
    │     若 conf < 0.90 或 afs ≥ 0.7 → 完整验证（2~3 次搜索）
    │
    ├─→ [Step C-1] 构造验证查询
    │     query_1 = 基于关系类型模板构造（见 C.2）
    │     query_2 = "{s} {o} 关系" 或 "{s} {p} {o} 是否正确"
    │
    ├─→ [Step C-2] 调用 web_search 工具
    │     results = web_search(query_1, sources=["wikipedia.org", "baike.baidu.com"], top_k=3)
    │
    ├─→ [Step C-3] LLM 判断一致性
    │     Prompt:
    │     "已知三元组：({s}, {p}, {o})
    │      以下是搜索结果：{results.snippets}
    │      请判断该三元组是否与搜索结果一致？
    │      输出：{verdict: SUPPORT|CONTRADICT|INSUFFICIENT, reason: str, correction: str|null}"
    │
    ├─→ [Step C-4] 根据验证结果处理
    │     ├─ SUPPORT     → verification_status = "VERIFIED"，conf 提升 +0.05
    │     ├─ CONTRADICT  → 进入修正流程（见 C.4）
    │     └─ INSUFFICIENT→ verification_status = "LOW_CONF"，保留并标注
    │
    └─→ 输出带验证状态的三元组
```

#### C.4 自动修正流程

```
当验证结果为 CONTRADICT 时：

1. 提取 correction 字段中的修正建议
2. 尝试自动修正：
   ├─ 若 object 错误 → 用搜索结果中的正确值替换
   ├─ 若 predicate 错误 → 重新归一化谓词
   └─ 若 subject 错误 → 标记为 DISCARD（主体错误无法自动修正）
3. 修正后重新调用 web_search 验证（最多 2 次修正循环）
4. 若修正后仍为 CONTRADICT → 标记为 DISCARD，从图谱中移除
5. 若修正成功 → 标记为 VERIFIED_CORRECTED，记录原始值和修正值
```

#### C.5 验证结果汇总与图谱入库

```
三元组最终状态分类：

VERIFIED          → 直接入图，高权重边
VERIFIED_CORRECTED→ 入图，记录修正历史
LOW_CONF          → 入图，低权重边，后续类比挖掘权重降低
DISCARD           → 不入图，记录到错误日志

知识图谱构建：
├─ 节点（实体）：
│   V = {entity_name, entity_type, domain, degree_centrality, embedding}
├─ 边（关系）：
│   E = {subject, predicate, object, relation_type, confidence,
│         afs, verification_status, source_doc}
└─ 图结构分析：
    ├─ 社区检测（Louvain 算法）→ 识别语义实体簇
    ├─ 节点中心性计算（PageRank + Betweenness）
    └─ 跨社区边识别（类比候选的重要线索）

输出：
├─ 知识图谱文件 G（Neo4j dump 或 NetworkX GraphML）
├─ 三元组置信度表（CSV）
└─ 构建统计报告：
    {
      "total_docs_processed": N,
      "total_triples_extracted": n_ext,
      "triples_verified": n_ver,
      "triples_corrected": n_cor,
      "triples_discarded": n_dis,
      "triples_low_conf": n_low,
      "graph_nodes": n_nodes,
      "graph_edges": n_edges,
      "avg_afs": float,
      "domain_distribution": {...}
    }
```

---

### Step-B/C 联合迭代机制

```
为保证图谱质量，Step-B 和 Step-C 支持联合迭代：

┌─────────────────────────────────────────────────────┐
│                                                     │
│  Step-B 批量抽取（batch_size=100 条三元组）           │
│       │                                             │
│       ▼                                             │
│  Step-C 批量验证                                    │
│       │                                             │
│       ├─ 修正信息反馈给 Step-B                       │
│       │   └─ 更新 LLM 提示词中的"常见错误模式"        │
│       │       示例：发现 LLM 经常把"X 导致 Y"抽取     │
│       │             为双向关系 → 在 Prompt 中强调     │
│       │             因果关系的方向性                  │
│       │                                             │
│       └─ 每处理 500 条三元组，输出中间图谱快照         │
│           供人工抽检（随机抽取 20 条审核）             │
└─────────────────────────────────────────────────────┘
```

---

## Phase 1: Analogy pattern mining

### 1.1 一阶类比候选识别 Agent

**输入**：知识图谱 `G`，实体集合 `V`

**方法**：
1. **属性向量化**：为每个实体构建属性特征向量
2. **语义相似度计算**：使用嵌入模型（如 GTE、BGE）计算实体对相似度
3. **结构相似度计算**：比较实体的邻居结构（Jaccard 相似度、图编辑距离）
4. **候选对生成**：阈值过滤后输出候选类比对 `{(Aᵢ, Bᵢ, sim_score)}`

**筛选标准**：
- 语义相似度 > 0.6，但不完全相同（避免同义词）
- 至少共享 2 个以上相同属性类型
- 属于不同领域或上位类（跨域类比优先）

**输出**：一阶类比候选对集合 `AC₁ = {(A, B, score, evidence)}`

---

### 1.2 二阶类比候选识别 Agent

**输入**：知识图谱 `G`，关系集合

**方法**：
1. **关系模式抽取**：提取关系四元组 `(A, r, B)` vs `(C, r', D)` 的结构模式
2. **关系同构检测**：判断两组关系是否满足 `r(A,B) ≅ r'(C,D)` 的映射关系
3. **类比链构造**：`A:B :: C:D`（即 A之于B，如同C之于D）
4. **类比强度评分**：综合考虑关系语义相似性、实体属性匹配度

**典型模式**：

```
功能类比：心脏:泵血 :: CPU:运算
层级类比：国王:国家 :: 总裁:公司
因果类比：摩擦:热量 :: 电流:磁场
组成类比：氢:水 :: 碳:有机物
```

**输出**：二阶类比候选集合 `AC₂ = {(A:B::C:D, pattern_type, score)}`

---

### 1.3 三阶类比候选识别 Agent

**输入**：知识图谱 `G`，社区检测结果

**方法**：
1. **子图采样**：基于社区结构和随机游走采样 K 跳子图
2. **子图同构检测**：使用图神经网络（GNN）或 Weisfeiler-Leman 算法检测子图结构相似性
3. **语义一致性验证**：确保结构同构的子图在语义上也具有可解释的类比关系
4. **类比映射构造**：建立两子图间节点和边的对应映射表 `φ: G₁→G₂`

**子图类比示例**：
```
太阳系子图：太阳-[中心]-行星-[围绕]-轨道-[决定]-引力
原子结构子图：原子核-[中心]-电子-[围绕]-轨道-[决定]-电磁力
⟹ 太阳系结构 ≅ 原子结构（玻尔模型）
```

**输出**：三阶类比候选集合 `AC₃ = {(SubG₁, SubG₂, mapping_φ, score)}`

---

## Phase 2: Data synthesis Agent (Agent A)

### 2.1 Query 生成策略

#### 显式类比 Query 模板

显式类比中，问题**直接询问**类比关系，模型需识别并解释类比。

**一阶显式 Query 模板**：
```
- "{A} 和 {B} 之间有什么相似之处？"
- "{A} 与 {B} 在 {属性维度} 上有何类比关系？"
- "从 {属性} 的角度看，{A} 就像 {B}，请解释这种类比。"
- "请比较 {A} 和 {B} 的相似性。"
```

**二阶显式 Query 模板**：
```
- "{A} 之于 {B}，如同 ___ 之于 {D}？（填空）"
- "{A}:{B} :: {C}:? 请完成这个类比。"
- "如果 {A} 对 {B} 的关系是 {rel1}，那么与之类比的关系是什么？"
- "请找出与"{A} 和 {B} 的关系"类似的另一对概念。"
```

**三阶显式 Query 模板**：
```
- "{系统/结构A} 和 {系统/结构B} 在结构上有何类比？"
- "请解释为什么 {领域A} 中的 {概念集合} 可以与 {领域B} 中的 {概念集合} 进行类比。"
- "{子图A描述} 的运作模式与 {子图B描述} 有何结构上的相似？"
```

#### 隐式类比 Query 模板

隐式类比中，问题**不直接提及类比**，但期望在 `think` 或 `answer` 中**运用类比推理**。

**一阶隐式 Query 模板**：
```
- "请描述 {A} 的工作原理。"（答案中用 B 进行类比解释）
- "{A} 的核心特征是什么？"（通过与 B 对比来揭示）
- "为什么说 {A} 具有 {某属性}？"（通过类比 B 来论证）
```

**二阶隐式 Query 模板**：
```
- "{A} 和 {B} 之间是什么关系？"（think 中运用 C:D 的类比来辅助理解）
- "如何理解 {A} 对 {B} 的影响？"（通过类比已知关系 C→D 推理）
- "请解释 {A} 为何导致 {B}。"（运用因果类比辅助论证）
```

**三阶隐式 Query 模板**：
```
- "请解释 {复杂系统/现象A} 的运作机制。"（think 中用熟悉的系统B做类比脚手架）
- "{领域问题A} 的解决方案是什么？"（迁移 {领域B} 中类似问题的解法）
- "如何向非专业人士解释 {抽象概念A}？"（通过子图类比进行通俗化解释）
```

---

### 2.2 Think 链构造规范

`think` 字段应体现**链式推理（Chain-of-Thought）**过程，结构如下：

```
[步骤1：识别已知量]
  - 明确问题中涉及的实体/关系/结构
  - 提取关键属性和约束条件

[步骤2：激活类比知识]（核心步骤）
  - 一阶：识别 A 与 B 的共同属性维度
  - 二阶：识别 r(A,B) 与 r'(C,D) 的关系结构同构
  - 三阶：识别 SubG₁ 与 SubG₂ 的拓扑映射

[步骤3：类比迁移推理]
  - 将已知侧（source domain）的知识迁移到目标侧（target domain）
  - 验证迁移的合法性（相似点 vs 差异点）

[步骤4：推导结论]
  - 基于类比得出结论
  - 标注类比的局限性（若有）
```

---

### 2.3 Answer 生成规范

`answer` 字段应满足：
1. **事实准确**：所有断言均可追溯到知识图谱中的三元组证据
2. **类比清晰**：明确指出类比的对应关系（source → target 映射）
3. **层次分明**：先给出直接答案，再展开类比解释
4. **局限说明**：在适当情况下指出类比的边界条件

---

### 2.4 合成执行流程

```
对每个类比候选 AC ∈ {AC₁ ∪ AC₂ ∪ AC₃}：
    │
    ├─→ 生成显式 Query（从对应模板池中采样）
    │       │
    │       └─→ 调用 LLM 生成 (think, answer)
    │
    └─→ 生成隐式 Query（从对应模板池中采样）
            │
            └─→ 调用 LLM 生成 (think, answer)
                确保 think 或 answer 中包含类比推理链
```

**LLM 提示词结构（System Prompt）**：
```
你是一个专业的类比推理数据合成专家。
给定知识图谱三元组证据，请生成高质量的类比推理问答数据。

要求：
1. think 字段须展示完整的类比推理链（CoT格式）
2. answer 字段须准确、简洁，有明确的类比对应关系
3. 确保类比在逻辑上是合理的、可验证的
4. 区分类比的相似点和本质差异

类比候选信息：
- 类比阶次：{order}
- Source Domain：{source_info}
- Target Domain：{target_info}
- 类比映射：{mapping}
- 支撑证据三元组：{evidence_triples}
```

**输出**：初始合成数据集 `D_raw = {(query, think, answer, order, analogy_type)}`

---

## Phase 3: Quality validation and correction Agent (Agent B)

### 3.1 验证维度

Agent B 对每条数据 `(query, think, answer)` 从以下维度进行自动验证：

| 验证维度 | 检查项 | 判定标准 |
|----------|--------|----------|
| **事实性** | answer 中的断言是否与知识图谱一致 | 三元组可溯源率 ≥ 80% |
| **类比合理性** | 类比映射是否在逻辑上成立 | 无明显逻辑谬误 |
| **推理完整性** | think 链是否包含完整的类比推理步骤 | 包含激活→迁移→验证三步 |
| **Query 合规性** | 显式/隐式类比标注是否正确 | 人工抽样一致率 ≥ 90% |
| **语言质量** | 表达是否流畅、无歧义 | 困惑度阈值检测 |
| **冗余性** | 是否与已有数据高度重复 | 语义相似度 < 0.85 |

---

### 3.2 验证执行流程

```
对每条数据 d ∈ D_raw：
    │
    ├─→ [检查1] 事实一致性验证
    │     - 提取 answer 中的断言
    │     - 在知识图谱中验证每条断言
    │     - 计算三元组支撑率
    │
    ├─→ [检查2] 类比逻辑验证（LLM-as-Judge）
    │     - Prompt: "以下类比是否在逻辑上成立？指出其合理性和潜在问题。"
    │     - 输出：{valid: bool, issues: [str], confidence: float}
    │
    ├─→ [检查3] 推理链完整性检测
    │     - 检测 think 字段是否包含必要推理步骤
    │     - 使用规则匹配 + LLM 双重验证
    │
    ├─→ [检查4] 重复性检测
    │     - 计算与数据池中已有数据的语义相似度
    │     - 过滤重复或高度相似的数据
    │
    └─→ 综合判定：
          ├─ PASS → 进入 D_validated
          ├─ MINOR_ISSUE → 自动修正后重新验证
          └─ MAJOR_ISSUE → 标记并触发重新生成
```

---

### 3.3 自动修正策略

| 问题类型 | 修正方法 |
|----------|----------|
| 事实错误 | 从知识图谱中替换错误断言 |
| 类比不清晰 | 补充显式的对应关系说明 |
| 推理链缺失步骤 | 插入缺失的中间推理步骤 |
| 表达不流畅 | LLM 重写（保持内容不变） |
| Query 模糊 | 从模板池重新采样并重新生成 |

**修正循环上限**：每条数据最多修正 3 次；超过上限仍不通过则丢弃。

**输出**：验证后数据集 `D_validated`，附带每条数据的验证报告

---

## Phase 4: Bloom classification and annotation Agent (Agent C)

### 4.1 分类目标

为 `D_validated` 中每条数据分配：
1. **Bloom 层次标签**：`Factual` 或 `Conceptual`
2. **细粒度类比类型标签**：从 12 种类型中选择（可多标签）
3. **类比阶次标签**：`1st-order`、`2nd-order`、`3rd-order`

---

### 4.2 分类规则

#### Factual vs Conceptual 判定

| 判定依据 | Factual（事实性） | Conceptual（概念性） |
|----------|-------------------|----------------------|
| 类比对象 | 具体实体、可测量属性、时间/空间/数量 | 抽象关系、系统结构、功能、因果模式 |
| 知识来源 | 直接可观测/记录的事实 | 需要理解概念关系的推断 |
| 验证方式 | 查阅百科/数据库可验证 | 需要逻辑推理/建模才能验证 |

#### 细粒度类型分类规则

```
Factual 类型：
├─ 实体类比：query/answer 以具体实体为比较对象
├─ 属性类比：比较维度为实体的可测量/可描述属性
├─ 时间类比：涉及时间顺序、持续时长、历史周期
├─ 空间类比：涉及位置、结构、空间布局关系
├─ 量化类比：涉及数量、比例、度量单位
└─ 事件类比：以事件（发生、因果、序列）为比较核心

Conceptual 类型：
├─ 关系类比：比较不同领域中概念间的关系模式
├─ 系统类比：比较复杂系统的整体结构和行为
├─ 功能类比：比较不同对象完成相似功能的方式
├─ 因果类比：类比两个领域中相似的因果链
├─ 比例类比：比较抽象的比例或规律关系
└─ 隐喻/比喻类比：以比喻或象征方式建立抽象类比
```

---

### 4.3 分类执行流程

```
对每条数据 d ∈ D_validated：
    │
    ├─→ [Step 1] 自动分类（LLM 分类器）
    │     Prompt:
    │     "根据以下分类规则，对给定的类比数据进行分类。
    │      输出：{bloom_level, analogy_types[], confidence}"
    │
    ├─→ [Step 2] 规则验证器（Rule-Based Checker）
    │     - 检查分类与类比阶次的一致性
    │     - 一阶类比 → 偏向 Factual（但可以是 Conceptual）
    │     - 二阶类比 → 两者均可
    │     - 三阶类比 → 偏向 Conceptual
    │
    ├─→ [Step 3] 置信度检查
    │     - confidence ≥ 0.8 → 直接采纳
    │     - 0.6 ≤ confidence < 0.8 → 第二次 LLM 验证
    │     - confidence < 0.6 → 标记为"需人工审核"
    │
    └─→ [Step 4] 标签写入
          - 将 (bloom_level, analogy_types, order, explicit/implicit)
            写入数据条目
```

---

### 4.4 分类验证与统计

**自动统计报告**：
```json
{
  "total_samples": N,
  "by_order": {
    "1st_order": n1,
    "2nd_order": n2,
    "3rd_order": n3
  },
  "by_bloom": {
    "Factual": {
      "entity_analogy": n,
      "attribute_analogy": n,
      "temporal_analogy": n,
      "spatial_analogy": n,
      "quantitative_analogy": n,
      "event_analogy": n
    },
    "Conceptual": {
      "relational_analogy": n,
      "system_analogy": n,
      "functional_analogy": n,
      "causal_analogy": n,
      "proportional_analogy": n,
      "metaphorical_analogy": n
    }
  },
  "by_query_type": {
    "explicit": n_exp,
    "implicit": n_imp
  },
  "flagged_for_review": n_flag
}
```

**输出**：最终带标注数据集 `D_final`，格式如下：

```json
{
  "id": "analogy_001",
  "query": "...",
  "think": "...",
  "answer": "...",
  "metadata": {
    "order": "2nd_order",
    "query_type": "explicit",
    "bloom_level": "Conceptual",
    "analogy_types": ["relational_analogy", "functional_analogy"],
    "source_domain": "...",
    "target_domain": "...",
    "evidence_triples": [["A", "rel", "B"], ["C", "rel'", "D"]],
    "confidence": 0.92,
    "validation_status": "PASS"
  }
}
```

---

## Full-pipeline Agent collaboration

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          Orchestrator Agent                                │
│                     （任务调度、状态管理、全局异常处理）                      │
└────────────────────────────┬───────────────────────────────────────────────┘
                             │
            ╔════════════════▼═══════════════════════════════╗
            ║       Sub-Module 0: KG Builder 子工作流         ║
            ║                                                 ║
            ║  ┌──────────┐   ┌──────────┐   ┌───────────┐  ║
            ║  │ Step-A   │──►│ Step-B   │──►│ Step-C    │  ║
            ║  │ 数据收集  │   │ 三元组抽取│   │ Web验证   │  ║
            ║  │ HF下载   │   │ LLM+规则 │   │web_search │  ║
            ║  └──────────┘   └──────────┘   └─────┬─────┘  ║
            ║                      ▲    修正反馈    │        ║
            ║                      └───────────────┘        ║
            ║                                               ║
            ║  输出：知识图谱 G + 三元组置信度表              ║
            ╚════════════════╤══════════════════════════════╝
                             │
                             ▼
                  ┌─────────────────────┐
                  │   类比模式挖掘 Agent  │
                  │                     │
                  │ - 一阶候选识别       │
                  │ - 二阶候选识别       │
                  │ - 三阶候选识别       │
                  └──────────┬──────────┘
                             │
            ╔════════════════▼═══════════════════════════════╗
            ║      Sub-Module 2: 数据合成子工作流 (Agent A)   ║
            ║                                                 ║
            ║  ┌──────────────┐    ┌──────────────────────┐  ║
            ║  │ Query 生成   │    │  Think/Answer 生成    │  ║
            ║  │ 显式 + 隐式  │───►│  CoT 链式推理         │  ║
            ║  └──────────────┘    └──────────────────────┘  ║
            ║                                               ║
            ║  输出：D_raw = {(query, think, answer)}        ║
            ╚════════════════╤══════════════════════════════╝
                             │
            ╔════════════════▼═══════════════════════════════╗
            ║      Sub-Module 3: 质量验证子工作流 (Agent B)   ║
            ║                                                 ║
            ║  ┌──────────┐  ┌──────────┐  ┌────────────┐   ║
            ║  │ 事实验证  │  │ 逻辑验证  │  │ 完整性验证 │   ║
            ║  └──────────┘  └──────────┘  └────────────┘   ║
            ║       │                           │            ║
            ║       └──────────┬────────────────┘            ║
            ║                  ▼                             ║
            ║         ┌────────────────┐                    ║
            ║         │  自动修正循环  │◄── 最多 3 次        ║
            ║         └────────────────┘                    ║
            ║                                               ║
            ║  输出：D_validated                             ║
            ╚════════════════╤══════════════════════════════╝
                             │
            ╔════════════════▼═══════════════════════════════╗
            ║      Sub-Module 4: Bloom 分类子工作流 (Agent C) ║
            ║                                                 ║
            ║  ┌──────────────┐  ┌────────────┐  ┌────────┐  ║
            ║  │ LLM 自动分类  │─►│  规则验证   │─►│ 置信度 │  ║
            ║  └──────────────┘  └────────────┘  │  过滤  │  ║
            ║                                    └────────┘  ║
            ║  输出：D_final（带完整 Bloom + 阶次 + 类型标注） ║
            ╚════════════════════════════════════════════════╝
                             │
                             ▼
                    最终数据集 D_final
```

---

## Recommended tech stack

| 模块 | 推荐工具/框架 |
|------|---------------|
| **数据集下载** | `datasets`（HuggingFace）、`huggingface_hub` |
| **文档解析** | `PyMuPDF`、`python-docx`、`BeautifulSoup` |
| **NER & 共指消解** | `spaCy en_core_web_trf`、`GLiNER`、`FastCoref` |
| **关系抽取** | `DeepKE`、`REBEL`（基于 BART）、`LLM（Qwen2.5/GPT-4）` |
| **知识图谱存储** | `Neo4j`（图数据库）或 `NetworkX`（轻量级本地） |
| **实体嵌入** | `BGE-M3`、`GTE-large`、`text-embedding-3` |
| **子图同构检测** | `PyTorch Geometric`（GNN）、`networkx.algorithms.isomorphism` |
| **web_search 工具** | `Tavily API`（推荐）、`Bing Search API`、`DuckDuckGo Search`（免费）、`SerpAPI` |
| **LLM 推理** | `OpenAI GPT-4o`、`Qwen2.5-72B`、`DeepSeek-R1` |
| **Agent 框架** | `LangGraph`（推荐，支持循环图）、`AutoGen`、`CrewAI` |
| **数据存储** | `JSON Lines`（.jsonl）+ `SQLite`（元数据索引）|
| **验证评估** | `LLM-as-Judge` + 规则检查器 |

---

## Quality control targets

| 指标 | 目标值 |
|------|--------|
| 三元组抽取精确率 | ≥ 85% |
| 类比候选有效率 | ≥ 70% |
| 数据通过验证率（Phase 3） | ≥ 80% |
| Bloom 分类一致率（人工抽检） | ≥ 88% |
| 显式/隐式类比比例 | 约 4:6（隐式偏多） |
| 一/二/三阶类比比例 | 约 4:4:2（三阶较难） |
| 最终数据集规模（建议） | ≥ 5,000 条 |

---

## Appendix: Data examples

### 示例 1：一阶显式类比（Factual - 属性类比）

```json
{
  "query": "心脏和水泵在功能上有哪些相似之处？",
  "think": "步骤1：识别已知量——心脏是人体器官，水泵是机械装置。\n步骤2：激活类比——心脏推动血液循环，水泵推动液体流动，两者均具有推动流体定向流动的核心功能。\n步骤3：类比迁移——心脏的瓣膜对应水泵的止回阀（防止倒流），心脏的收缩对应水泵的活塞运动，心率对应泵的频率。\n步骤4：得出结论——心脏是一种生物液体泵，其功能结构与机械水泵高度类比。",
  "answer": "心脏与水泵的类比关系：心脏通过有节律的收缩（对应水泵的活塞）推动血液（对应流体）在血管（对应管道）中循环流动，心脏瓣膜（对应止回阀）确保血液单向流动，心率（对应泵频）调节输出量。",
  "metadata": {
    "order": "1st_order",
    "query_type": "explicit",
    "bloom_level": "Factual",
    "analogy_types": ["functional_analogy", "attribute_analogy"]
  }
}
```

### 示例 2：二阶隐式类比（Conceptual - 关系类比）

```json
{
  "query": "为什么说文字的发明对古代文明的传承至关重要？",
  "think": "步骤1：识别问题核心——文字与文明传承的关系。\n步骤2：激活类比——这类似于DNA与生物遗传信息传递的关系。DNA是生物遗传信息的载体，文字是文化信息的载体；DNA突变导致遗传信息改变，类比文字失传导致文化断层；DNA复制保证信息传递稳定，类比文字记录保证历史传承稳定。\n步骤3：迁移推理——从生物遗传的类比可知，文字作为"文化DNA"，使文明的知识、制度、价值观得以跨越时空传递，无需每代人重新探索。\n步骤4：结论——文字是文明延续的物质基础，其重要性类比于DNA对物种延续的意义。",
  "answer": "文字之于文明，如同DNA之于生命——文字是人类文化信息的存储和传递载体。没有文字，每代人的知识积累将随人的消逝而消亡，文明便无法实现累积性发展。文字的发明使得法律、技术、历史等跨代传递成为可能，是文明延续的核心机制。",
  "metadata": {
    "order": "2nd_order",
    "query_type": "implicit",
    "bloom_level": "Conceptual",
    "analogy_types": ["relational_analogy", "functional_analogy"]
  }
}
```

### 示例 3：三阶显式类比（Conceptual - 系统类比）

```json
{
  "query": "太阳系的结构与原子的玻尔模型在结构上有哪些对应关系？请进行详细的系统类比分析。",
  "think": "步骤1：识别两个系统——太阳系（宏观天体系统）与玻尔原子模型（微观量子系统）。\n步骤2：激活子图类比——\n  太阳系子图：太阳(大质量中心体)→[引力束缚]→行星→[轨道运动]→固定椭圆轨道\n  原子子图：原子核(带正电中心体)→[电磁力束缚]→电子→[轨道运动]→固定量子轨道\n步骤3：映射关系——\n  太阳 ↔ 原子核（中心质量/电荷来源）\n  行星 ↔ 电子（围绕中心运动的粒子）\n  引力 ↔ 电磁力（束缚力）\n  轨道周期 ↔ 量子数（决定轨道特征的参数）\n步骤4：验证与局限——类比成立的范围：拓扑结构相似；局限：行星轨道连续，电子轨道量子化，不可直接类推。",
  "answer": "太阳系与玻尔原子模型的系统类比：\n| 太阳系 | 原子（玻尔模型）| 对应关系 |\n|--------|----------------|----------|\n| 太阳 | 原子核 | 质量/能量中心，提供束缚力 |\n| 行星 | 电子 | 围绕中心运动的客体 |\n| 万有引力 | 库仑电磁力 | 向心束缚力 |\n| 轨道半径 | 量子轨道半径 | 决定运动路径 |\n| 行星质量 | 电子质量（极小）| 远小于中心体 |\n两者均为"中心力场+轨道运动"结构，这一类比启发了玻尔提出原子的行星模型。",
  "metadata": {
    "order": "3rd_order",
    "query_type": "explicit",
    "bloom_level": "Conceptual",
    "analogy_types": ["system_analogy", "functional_analogy"]
  }
}
```

---

## Implementation

This workflow is implemented in the repo as follows. The two docs are aligned in content (Chinese: `agent_workflow_CN.md`, English: `agent_workflow.md`).

| Phase | Directory / entry | Description |
|-------|-------------------|-------------|
| **Sub-Module 0** KG build | `AnalogyKG/` | Corpus prep, analogy triple extraction (entity/relation/subgraph), web verification and correction; entry `build.py`, config and prompts under `config/`. |
| **Sub-Modules 2–4** instruction synthesis & validation | `AnalogySyn/` | Read analogy data → synthesize LLM training instructions (input, thinking, output) → Bloom classification → validate → correct if invalid; entry `run.py`, with logic in `synthesize.py`, `validate.py`, `correct.py`; config and prompts under `config/`. |

Data flow: `AnalogyKG` writes `entity/relation/subgraph_analogy_triples.jsonl` to `data_dir`; `AnalogySyn` reads from that directory and produces fine-tuning instructions `instructions.jsonl`.

---

*Doc version: v1.0 | Date: 2026-03-12*
