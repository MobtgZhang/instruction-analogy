# Task-analogy

面向**类比推理**的 LLM 微调数据合成：从文档与类比三元组生成 `(input, thinking, output)` 形式的训练样本，并打上 Bloom 认知分类标签。

## 原理概要

1. **知识图谱构建**  
   从文档语料（如 Wikipedia）中抽取**类比导向的三元组**（实体类比、关系类比、子图类比），经 Web 检索验证与修正，输出高质量三元组：`entity_analogy_triples.jsonl`、`relation_analogy_triples.jsonl`、`subgraph_analogy_triples.jsonl`。

2. **指令合成**  
   对每条类比记录，**合成**一条 LLM 训练样本：用户 **input**（问题）、模型 **thinking**（链式类比推理）、**output**（答案）；可结合在线百科等信息。并对样本做 **Bloom 分类**（Factual/Conceptual 及 12 种子类型）。

3. **验证与修正**  
   对每条合成指令进行**验证**（事实一致、推理完整、表达清晰）；若不通过则根据原始类比依据**修正**后重新验证，直至通过或达到重试上限。

4. **输出**  
   通过验证的指令写入 `instructions.jsonl`，字段含 `input`、`thinking`、`output`、`bloom_level`、`bloom_types`。

完整工作流（Agent 划分、Prompt、数据格式、质量指标）见 **`agent_workflow.md`**（英文）与 **`agent_workflow_CN.md`**（中文），两处内容一致。

## 目录说明

| 路径 | 作用 |
|------|------|
| `AnalogyKG/` | 知识图谱构建：语料准备、三元组抽取、Web 验证与修正（`build.py`、`config/`）。 |
| `AnalogySyn/` | 指令合成：读取三元组 → 合成 → 分类 → 验证 → 修正（`run.py`、`synthesize.py`、`validate.py`、`correct.py`、`config/`）。 |
| `agent_workflow.md` | 工作流完整说明（英文）。 |
| `agent_workflow_CN.md` | 工作流完整说明（中文）。 |

## 快速运行

- **AnalogyKG**：在 `data/balanced/` 下准备语料，配置 `config/api.json`（如 `openai_api_key`），执行 `python -m AnalogyKG.build` 或 `cd AnalogyKG && python build.py`。
- **AnalogySyn**：配置 `AnalogySyn/config/api.json` 或环境变量 `OPENAI_API_KEY`，将 `data_dir` 指向存放三元组 `.jsonl` 的目录，执行 `python -m AnalogySyn.run` 或 `cd AnalogySyn && python run.py`，输出在 `AnalogySyn/data/instructions.jsonl`。
