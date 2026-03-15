# Task-analogy

Synthesis of **analogy-reasoning datasets** for LLM fine-tuning. The pipeline turns documents and analogy triples into training samples of the form `(input, thinking, output)`, with Bloom taxonomy labels.

## Principle

1. **Knowledge graph (KG) build**  
   From a document corpus (e.g. Wikipedia), extract **analogy-oriented triples** (entity, relation, subgraph analogies), then verify and correct them with web search. Output: high-quality triples in `entity_analogy_triples.jsonl`, `relation_analogy_triples.jsonl`, `subgraph_analogy_triples.jsonl`.

2. **Instruction synthesis**  
   For each analogy record, **synthesize** an LLM training sample: user **input** (query), model **thinking** (chain-of-thought analogy reasoning), and **output** (answer). Optional online encyclopedia context can be used. Samples are **classified** into Bloom categories (Factual / Conceptual and 12 fine-grained types).

3. **Validation and correction**  
   Each synthesized instruction is **validated** (factual consistency, reasoning completeness, clarity). If invalid, it is **corrected** using the same analogy evidence and re-validated until pass or a retry limit is reached.

4. **Output**  
   Valid instructions are written to `instructions.jsonl` with fields: `input`, `thinking`, `output`, `bloom_level`, `bloom_types`.

Detailed workflow (agents, prompts, data formats, quality targets) is in **`agent_workflow.md`** (English) and **`agent_workflow_CN.md`** (Chinese); the two files are aligned in content.

## Repo layout

| Path | Role |
|------|------|
| `AnalogyKG/` | KG build: corpus, triple extraction, web verification & correction (`build.py`, `config/`). |
| `AnalogySyn/` | Instruction synthesis: read triples → synthesize → classify → validate → correct (`run.py`, `synthesize.py`, `validate.py`, `correct.py`, `config/`). |
| `agent_workflow.md` | Full workflow spec (English). |
| `agent_workflow_CN.md` | Full workflow spec (Chinese). |

## Quick run

- **AnalogyKG**: prepare corpus under `data/balanced/`, set `config/api.json` (e.g. `openai_api_key`), then run `python -m AnalogyKG.build` (or `cd AnalogyKG && python build.py`).
- **AnalogySyn**: set `AnalogySyn/config/api.json` (or `OPENAI_API_KEY`), point `data_dir` to the directory containing the triple `.jsonl` files, then run `python -m AnalogySyn.run` (or `cd AnalogySyn && python run.py`). Output: `AnalogySyn/data/instructions.jsonl`.
