# Router — LangGraph + Ollama

A small collection of examples showing how to:
- Route user intent across multiple Ollama models (LangGraph + LangChain)
- Use a Streamlit chat UI with routing and time-bounded streaming
- Expose chains and models via FastAPI + LangServe
- Generate simple observability mock data (and a LangSmith demo)

The code is split into focused scripts so you can pick what you need.

## Prerequisites
- Python 3.10+
- Ollama running locally (default base URL `http://localhost:11434`)
  - Install: https://ollama.com
  - Pull or replace the models used in the code (see the model names in the files). You can change them to any local models you have.

## Setup
```bash
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Optional environment variables:
- `OLLAMA_BASE_URL` — override Ollama URL (default `http://localhost:11434`).
- `LANGSMITH_API_KEY` — required only for LangSmith tracing in `observ_langsmith.py`.

## Apps and Scripts

- Streamlit Router UI: rich chat with routing and soft timeouts
  - Run: `streamlit run app.py`
  - Notes: configure temperature, streaming, HTTP timeout, and system prompt in the sidebar.
  - New: select `Web/Tools agent` mode to enable tools (web_search, web_get, calculator, read_file). In `Auto (router)` mode, messages containing URLs or web keywords will auto-route to tools.

- Minimal Streamlit chat example
  - Run: `streamlit run app_streamlit.py`

- FastAPI + LangServe server
  - Full demo: `uvicorn server_langserve:app --reload --port 8000`
    - Open: http://127.0.0.1:8000/docs and try `/ollama`, `/essay`, `/song`
  - Minimal sanity server: `uvicorn sanity_langserve:app --reload --port 8080`
    - Open: http://127.0.0.1:8080/ and `/docs` or `/ollama/playground/`

- LangSmith observability demo (optional)
  - Requirements: set `LANGSMITH_API_KEY` and have LangSmith project configured
  - Run: `python observ_langsmith.py`
  - Track in UI: https://smith.langchain.com (open Project "Router-App")

### LangSmith Dashboard
- UI: https://smith.langchain.com
- Project: set `LANGCHAIN_PROJECT=Router-App` (see examples below)
- Filter runs by tags from examples, e.g. `suite=chat-smoke`, `suite=rag-batch`

## LangSmith Tests (PowerShell)

The snippets below are ready to paste into Windows PowerShell. They enable tracing, run a few smoke tests, and optionally seed a small dataset in LangSmith you can use in the UI.

- Environment and tracing
  - Set once per session:

```
$env:LANGSMITH_API_KEY = "<your-key>"
$env:LANGCHAIN_TRACING_V2 = "true"
$env:LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
$env:LANGCHAIN_PROJECT = "Router-App"
```

- Chat smoke test (traced to LangSmith)

```
python .\observ_langsmith.py --enable --project "Router-App" --tags "suite=chat-smoke,model" --prompt "Give a 1-sentence summary of retrieval-augmented generation."
```

- RAG smoke test (traced to LangSmith)

```
python .\observ_langsmith.py rag --enable --project "Router-App" --tags "suite=rag-smoke" --question "What does the uploaded document say about risks?" --k 4
```

- Run a small chat test suite

```
$tests = @(
  @{ Prompt = "Explain LangGraph in one sentence"; Tags = "suite=chat-batch,topic=langgraph" },
  @{ Prompt = "List 3 common RAG pitfalls";       Tags = "suite=chat-batch,topic=rag" },
  @{ Prompt = "Summarize the benefits of Ollama"; Tags = "suite=chat-batch,topic=ollama" }
)

foreach ($t in $tests) {
  python .\observ_langsmith.py --enable --project "Router-App" --tags $t.Tags --prompt $t.Prompt
}
```

- Run a small RAG test suite

```
$questions = @(
  "What sections mention limitations?",
  "Summarize key risks in two bullets",
  "Where does the document discuss evaluation?"
)

foreach ($q in $questions) {
  python .\observ_langsmith.py rag --enable --project "Router-App" --tags "suite=rag-batch" --question $q --k 5
}
```

- Create a tiny dataset in LangSmith (optional)
  - After creating it, open LangSmith > Datasets > `Router-QuickQA` to review or run evaluations in the UI.

```
$code = @'
from langsmith import Client

client = Client()
dataset = client.create_dataset(
    dataset_name="Router-QuickQA",
    description="Quick QA set for Router demo",
)

client.create_example(
    inputs={"prompt": "Explain LangGraph in one sentence"},
    outputs={"expected": "A stateful framework for building multi-step LLM workflows."},
    dataset_id=dataset.id,
)

client.create_example(
    inputs={"prompt": "List 3 common RAG pitfalls"},
    outputs={"expected": "Shallow retrieval, hallucinated synthesis, and poor chunking."},
    dataset_id=dataset.id,
)

print(f"Created dataset: {dataset.name} ({dataset.id})")
'@

Set-Content -Path .\tmp_langsmith_ds.py -Value $code
python .\tmp_langsmith_ds.py
Remove-Item .\tmp_langsmith_ds.py
```

### Dataset Evals (PowerShell)
- Prereq: create the dataset first (see snippet above) or set `$env:LS_DATASET` to an existing dataset name in LangSmith.

```
# Required env
$env:LANGSMITH_API_KEY = "<your-key>"
$env:LANGCHAIN_TRACING_V2 = "true"
$env:LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
$env:LANGCHAIN_PROJECT = "Router-App"
$env:LS_DATASET = "Router-QuickQA"

# Models to compare (edit as needed)
$models = @(
  "mistral:7b-instruct-v0.3-q5_0",
  "llama3.1:8b-instruct-q4_0"
)

# Tiny evaluator runner: builds a prompt+model chain and runs it over the dataset
$code = @'
from langsmith import Client
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import os

dataset_name = os.environ.get("LS_DATASET", "Router-QuickQA")
project = os.environ.get("LANGCHAIN_PROJECT", "Router-App")
model = os.environ.get("OLLAMA_MODEL", "mistral:7b-instruct-v0.3-q5_0")
base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

def chain_factory():
    try:
        llm = ChatOllama(model=model, base_url=base_url, temperature=0, request_timeout=1800)
    except TypeError:
        llm = ChatOllama(model=model, base_url=base_url, temperature=0, timeout=1800)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise helpful assistant. Answer briefly."),
        ("human", "{prompt}")
    ])
    return prompt | llm

client = Client()
client.run_on_dataset(
    dataset_name=dataset_name,
    llm_or_chain_factory=chain_factory,
    project_name=f"{project} - DatasetEval - {model}",
    tags=[f"model={model}", "suite=dataset-eval"],
)
print(f"Started dataset eval for {model}. Check LangSmith UI.")
'@

Set-Content -Path .\tmp_dataset_eval.py -Value $code

foreach ($m in $models) {
  $env:OLLAMA_MODEL = $m
  python .\tmp_dataset_eval.py
}

Remove-Item .\tmp_dataset_eval.py
```

Notes
- The examples above tag runs (e.g., `suite=chat-smoke`) so you can filter in LangSmith.
- Change the `--model` and `--base-url` flags in the commands if you want to test different local Ollama models or endpoints.
- For larger test suites or automated CI, keep the here-string scripts in version control and call them from your pipeline.

## Model Names
The examples reference several models, e.g. `mixtral:8x7b-instruct-v0.1-q4_0`, `deepseek-v2.5`, `llama3.1:8b-instruct-q4_0`, `llama3.3:70b-instruct-q4_K_M`, etc. If you don’t have these models locally, either:
- Pull them with `ollama pull <model-name>`, or
- Replace the names in the scripts with models you already have.

Key files to review:
- `app.py` — Streamlit chat router with soft timeout after first token
- `agents/tools_agent.py` — ReAct agent wiring with basic tools
- `tools/basic_tools.py` — Tools: now/today (czas i data), web fetch/search, calculator, file read
- `server_langserve.py` / `sanity_langserve.py` — FastAPI + LangServe endpoints
- `observ_langsmith.py` — observability demos

## Troubleshooting
- Timeouts on first call: large models may need warm-up. Increase the HTTP timeout in `app.py`’s sidebar.
- `langchain_ollama.ChatOllama` timeout arg name can differ by version; `app.py` handles both `request_timeout` and `timeout`.
- If imports fail, ensure versions meet minimums in `requirements.txt` and your Python is 3.10+.
- Web tools rely on outbound HTTP. In restricted environments, network calls will fail; switch back to standard chat or provide direct content.

## To add

- For OCR add Mistral OCR https://mistral.ai/news/mistral-ocr https://huggingface.co/spaces/merterbak/Mistral-OCR - seems to not be open-source
- Llava:34b looks promising
- Fix vision, reads not pdf, looks bad
