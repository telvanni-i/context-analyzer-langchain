# Context Analyzer (LangChain + LangGraph example)

This repository shows a **simple but production-style structure** for a LangChain app.

It implements the graph you requested:
1. Read a file containing the user request.
2. Read `context/JIRA.txt`.
3. Send both to an LLM and convert the task into a step-by-step algorithm where each step has:
   - `step_description`
   - `tags`
   - `embedding`

## Why this structure is best practice

This project is intentionally split into small folders so each concern is easy to understand and replace:

- `graph/`: orchestration only (node order and flow).
- `agents/`: business logic that talks to LLMs.
- `tools/`: reusable utilities (for example, file readers).
- `models/`: strict output schemas for reliable structured responses.
- `config/`: environment and secret loading.
- `utils/`: generic helpers like JSON writing.

Benefits:
- **Testability**: small units are easy to test in isolation.
- **Replaceability**: swap models, tools, or prompt logic without rewriting flow.
- **Readability**: beginners can inspect one file at a time.
- **Safety**: structured outputs reduce parser errors.

## Project tree

```text
.
├── context/
│   └── JIRA.txt
├── input/
│   └── user_request.txt
├── context_analyzer/
│   ├── agents/
│   │   ├── base.py
│   │   └── decomposition_agent.py
│   ├── config/
│   │   └── settings.py
│   ├── graph/
│   │   ├── nodes.py
│   │   ├── state.py
│   │   └── workflow.py
│   ├── models/
│   │   └── schemas.py
│   ├── tools/
│   │   ├── base.py
│   │   └── file_reader.py
│   ├── utils/
│   │   └── io.py
│   └── main.py
├── .env.local.example
└── pyproject.toml
```

## How it works (step by step)

### 1) Input node: read user request
`read_request_node` reads the request text file (default: `input/user_request.txt`).

### 2) Context node: read JIRA context
`read_jira_context_node` reads `context/JIRA.txt` so the model has extra project constraints and scope.

### 3) Decomposition node: LLM + embeddings
`decompose_task_node` runs `DecompositionAgent` which:
- prompts an OpenAI chat model for structured decomposition,
- validates output into `DecompositionResult` (`pydantic` model),
- computes embedding for each `step_description`,
- returns JSON-ready output.

## Setup

1. Create env file:

```bash
cp .env.local.example .env.local
```

2. Add your real API key in `.env.local`.

3. Install dependencies:

```bash
pip install -e .
```

## Run

```bash
python -m context_analyzer.main \
  --request-file input/user_request.txt \
  --output-file output/decomposition.json
```

## Example output format

```json
{
  "steps": [
    {
      "step_description": "Define project modules and interfaces.",
      "tags": ["architecture", "setup"],
      "embedding": [0.0132, -0.0091, 0.0043]
    }
  ]
}
```

## Notes

- The runtime reads credentials and logging settings from `.env.local`.
- Model defaults:
  - Chat model: `gpt-4.1-mini`
  - Embedding model: `text-embedding-3-small`
- Optional SOCKS5 proxy can be configured with `SOCKS5_URL` (used by both chat and embeddings clients).
- Every LLM interaction is appended to `OPENAI_LOGS_PATH` as JSONL.
- You can override model names via env vars (`OPENAI_MODEL`, `OPENAI_EMBEDDING_MODEL`).
