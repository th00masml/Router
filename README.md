# Router — LangGraph + Ollama with RAG

A comprehensive AI application framework that combines intent-based model routing, Retrieval Augmented Generation (RAG), and intelligent tool usage. This project demonstrates advanced LangChain patterns with local Ollama models.

## Key Features

- **Intelligent Model Routing**: Automatically routes queries to specialized Ollama models based on intent (chat, code, reasoning, long-context, RAG)
- **RAG System**: Complete document processing pipeline with vector storage using ChromaDB and Ollama embeddings
- **Agent Tools**: ReAct agent with web search, file reading, OCR, calculator, and time/date tools
- **Streamlit UI**: Rich chat interface with streaming responses, file upload, and document analysis
- **FastAPI Server**: REST API endpoints for model chains via LangServe
- **Observability**: Integrated LangSmith tracing and monitoring
- **Document Processing**: Support for text, markdown, HTML, JSON, CSV, and PDF files with OCR capabilities

## Architecture Overview

The Router project follows a modular architecture with clear separation of concerns:

```
Router/
├── app.py                 # Main Streamlit application
├── rag/
│   └── store.py          # Vector store operations (ChromaDB)
├── agents/
│   └── tools_agent.py    # ReAct agent implementation
├── tools/
│   └── basic_tools.py    # Tool definitions (web, OCR, calculator, etc.)
├── server_langserve.py   # FastAPI server with LangServe
├── observ_langsmith.py   # LangSmith observability utilities
├── rag_db/              # Persistent vector database
└── logs/                # Agent execution logs
```

## Prerequisites

### Required Software
- **Python 3.10+** with pip
- **Ollama** running locally (default: `http://localhost:11434`)
  - Install from: https://ollama.com
  - Ensure Ollama service is running

### Required Ollama Models

The application uses several specialized models. Pull them using:

```bash
# Core models (required)
ollama pull mistral:7b-instruct-v0.3-q5_0      # RAG worker
ollama pull llama3.1:8b-instruct-q4_0          # Light routing
ollama pull qwen3-coder:latest                 # Code assistance
ollama pull deepseek-v2.5                      # Reasoning tasks
ollama pull mistral-small3.2:24b               # Long context

# Embedding model (required for RAG)
ollama pull nomic-embed-text                   # Vector embeddings

# Vision model (optional, for OCR)
ollama pull llava:34b                          # Document OCR
```

**Note**: You can modify model names in `app.py` MODELS dictionary to use your preferred models.

## Installation

### 1. Clone and Setup Virtual Environment

```bash
git clone <repository-url>
cd Router

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows Command Prompt:
.venv\Scripts\activate.bat
# macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file (optional) or set environment variables:

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_OCR_MODEL=llava:34b
OLLAMA_TIMEOUT=180

# LangSmith Observability (optional)
LANGSMITH_API_KEY=your_langsmith_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=Router-App
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### 4. Verify Installation

Test Ollama connectivity:
```bash
curl http://localhost:11434/api/tags
```

## Usage

### 1. Streamlit Web Application (Primary Interface)

Launch the main application:

```bash
streamlit run app.py
```

The Streamlit interface provides:

- **Chat Interface**: Multi-model conversation with automatic intent routing
- **RAG Document Upload**: Upload and index documents for question-answering
- **File Analysis**: Analyze uploaded files with custom prompts
- **OCR/Document Reader**: Extract text from images and PDFs
- **Agent Tools**: Enable web search, file reading, calculator, and OCR tools
- **Configuration Panel**: Adjust model parameters, timeouts, and settings

#### Key Interface Features:

- **Automatic Routing**: Messages are routed to appropriate models based on content
- **Manual Mode Selection**: Override routing by selecting specific models
- **Streaming Responses**: Real-time token streaming with stop capability
- **Soft Timeouts**: Configurable response time limits
- **Per-Model Configuration**: Context windows and token limits
- **LangSmith Integration**: Optional observability and tracing

### 2. FastAPI Server (REST API)

Launch the API server:

```bash
uvicorn server_langserve:app --reload --port 8000
```

Access the interactive API docs at: http://localhost:8000/docs

Available endpoints:
- `/ollama` - Direct model chat
- `/essay` - Essay generation chain
- `/song` - Song writing chain

### 3. Command Line Observability

Run observability demos:

```bash
# Chat example with tracing
python observ_langsmith.py --enable --project "Router-App" --prompt "Explain RAG in one sentence"

# RAG example with tracing
python observ_langsmith.py rag --enable --question "What are the key features?" --k 5
```

## RAG (Retrieval Augmented Generation) System

The Router project includes a complete RAG implementation for document-based question answering.

### Vector Store Operations

The RAG system uses ChromaDB with Ollama embeddings:

```python
from rag.store import add_documents, similarity_search, delete_by_filename

# Add documents to vector store
add_documents(base_url, [
    {"text": "Document content...", "metadata": {"filename": "doc1.txt"}}
])

# Search for relevant passages
results = similarity_search(base_url, "query", k=5)

# Delete documents by filename
delete_by_filename(base_url, "doc1.txt")
```

### Supported Document Formats

- **Text**: `.txt`, `.md` (markdown)
- **Web**: `.html` (with tag cleaning)
- **Data**: `.json`, `.csv`
- **PDF**: Text extraction + form fields
- **Images**: OCR via vision models (PNG, JPG, WEBP, etc.)

### Document Processing Pipeline

1. **Text Extraction**: Format-specific content extraction
2. **Chunking**: Recursive character splitting (800 chars, 120 overlap)
3. **Embedding**: Ollama `nomic-embed-text` model
4. **Storage**: Persistent ChromaDB collection
5. **Retrieval**: Similarity search with metadata filtering

### RAG Workflow in Streamlit

1. **Upload Documents**: Use "RAG: Documents" expander
2. **Ask Questions**: Use "RAG: Ask" expander with optional filename filtering
3. **Grounded Responses**: Citations to source documents

## Agent Tools System

The Router includes a ReAct agent with intelligent tool selection:

### Available Tools

| Tool | Purpose | Input Example |
|------|---------|---------------|
| `web_search` | DuckDuckGo search | "AI news 2024" |
| `web_get` | Fetch webpage content | "https://example.com" |
| `calculator` | Math expressions | "2*(3+4)^2" |
| `read_file` | Local file reading | "data/config.json" |
| `ocr_image` | Image text extraction | "document.png" |
| `now` | Current date/time | "" (empty input) |
| `today` | Current date | "" (empty input) |
| `time_in` | Timezone conversion | "Europe/Warsaw" |

### Tool Selection Logic

The agent automatically selects relevant tools based on query content:

- **Date/Time queries** → `now`, `today`, `time_in`
- **Web searches** → `web_search`, `web_get`
- **Math expressions** → `calculator`
- **File references** → `read_file`
- **OCR requests** → `ocr_image`

### Agent Configuration

- **Max Iterations**: Limit tool usage steps (default: 10)
- **Strict Mode**: Fast execution for time/date queries
- **Verbose Logging**: Detailed step traces
- **Timeout Settings**: Web request timeouts

## Configuration Guide

### Model Configuration

Edit the `MODELS` dictionary in `app.py` to customize model assignments:

```python
MODELS: Dict[str, str] = {
    "chat":          "your-chat-model",
    "reasoning":     "your-reasoning-model",
    "code":          "your-code-model",
    "router_light":  "your-light-model",
    "long_context":  "your-long-context-model",
    "rag_worker":    "your-rag-model",
}
```

### Intent Routing Rules

The system automatically routes queries based on keywords:

- **Code**: "code", "python", "function", "api", "bug", "compile"
- **Long Context**: "pdf", "document", "analyze file", long text (>800 words)
- **Reasoning**: "prove", "step by step", "logic", "mathematics"
- **RAG**: "extract", "summarize", "bullet points", "parse"
- **Light**: "short", "tl;dr", "brief", "concise"
- **Default**: "chat" for general queries

### Vector Store Configuration

Customize RAG settings in `rag/store.py`:

```python
# Chunking parameters
chunk_size = 800        # Characters per chunk
chunk_overlap = 120     # Overlap between chunks

# Embedding model
embed_model = "nomic-embed-text"  # Ollama embedding model

# Database location
DB_DIR = "rag_db"       # Persistent storage directory
```

### OCR Configuration

Configure document OCR in the Streamlit interface:

- **Model**: Vision model for text extraction (default: `llava:34b`)
- **DPI**: Render quality for PDFs (120-600)
- **Enhancement**: Contrast/binarization for handwriting
- **Tiling**: Split large images for better small text recognition

## Observability and Monitoring

### LangSmith Integration

The Router project includes comprehensive LangSmith observability:

- **Automatic Tracing**: All LLM calls and agent steps
- **Custom Metadata**: Model parameters, tool usage, retrieval context
- **Run Tagging**: Organize traces by functionality
- **Dashboard**: https://smith.langchain.com

### Local Logging

Agent execution logs are saved to `logs/` directory:

```json
{
  "timestamp": "20241003_143022",
  "input": "What time is it in Tokyo?",
  "model_alias": "router_light",
  "settings": {...},
  "steps": [...],
  "output": "It is currently 11:30 PM in Tokyo."
}
```

## LangSmith Testing Examples (PowerShell)

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

## API Documentation

### Vector Store API

The `rag.store` module provides a complete vector store interface:

```python
# Document management
add_document(base_url, text, metadata, embed_model="nomic-embed-text")
add_documents(base_url, items, embed_model="nomic-embed-text")
delete_by_filename(base_url, filename, embed_model="nomic-embed-text")
reset_vectorstore()  # Clear all documents

# Search operations
similarity_search(base_url, query, k=5, embed_model="nomic-embed-text")
similarity_search_filtered(base_url, query, k=5, filename=None, embed_model="nomic-embed-text")
list_filenames(base_url, embed_model="nomic-embed-text")
```

### Agent Tools API

Create and configure ReAct agents:

```python
from agents.tools_agent import build_tools_agent, invoke_tools_agent
from tools.basic_tools import get_basic_tools, select_tools_for_query

# Build agent with custom configuration
executor = build_tools_agent(
    llm=llm,
    max_iterations=10,
    verbose=False,
    tools_override=select_tools_for_query("search for Python tutorials"),
    strict=False
)

# Execute agent with chat history
result = invoke_tools_agent(executor, chat_history, user_input, run_config)
```

### Model Router API

Intent-based routing functionality:

```python
# Route determination
route = route_intent("Fix this Python code")
# Returns: "code"

route = route_intent("Analyze this 50-page research paper")
# Returns: "long_context"

# Manual route resolution
eff_route = resolve_route(chat_history, chosen_mode="auto")
```

## File Structure Reference

### Core Application Files

- **`app.py`** — Main Streamlit application with chat UI, RAG interface, and tool integration
- **`lang.py`** — Simple Ollama model demonstration
- **`requirements.txt`** — Python dependencies specification

### RAG System

- **`rag/store.py`** — Vector store operations using ChromaDB and Ollama embeddings
- **`rag_db/`** — Persistent vector database storage

### Agent System

- **`agents/tools_agent.py`** — ReAct agent implementation with LangChain
- **`tools/basic_tools.py`** — Tool definitions for web, math, file, time, and OCR operations

### Server Components

- **`server_langserve.py`** — FastAPI server with LangServe chain endpoints
- **`observ_langsmith.py`** — LangSmith observability utilities and CLI testing

### Data and Logs

- **`logs/`** — Agent execution logs in JSON format
- **`prompts/examples.yaml`** — Example prompts organized by model type

## Model Requirements

The Router project is designed to work with various Ollama models. Here are the recommended models by category:

### Required Models

```bash
# Core functionality
ollama pull mistral:7b-instruct-v0.3-q5_0      # RAG and general tasks
ollama pull llama3.1:8b-instruct-q4_0          # Light routing
ollama pull nomic-embed-text                   # Embeddings (required for RAG)
```

### Optional Specialized Models

```bash
# Code assistance
ollama pull qwen3-coder:latest                 # Or codellama, deepseek-coder

# Reasoning tasks
ollama pull deepseek-v2.5                      # Or llama3.1:70b

# Long context
ollama pull mistral-small3.2:24b               # Or claude-3-haiku via OpenAI API

# Vision/OCR
ollama pull llava:34b                          # Or llava:13b for faster processing
```

### Model Customization

To use different models, update the `MODELS` dictionary in `app.py`:

```python
MODELS: Dict[str, str] = {
    "chat":          "llama3.1:8b-instruct-q4_0",  # Your preferred chat model
    "reasoning":     "llama3.1:70b-instruct-q4_0", # Your reasoning model
    "code":          "codellama:13b-instruct",      # Your code model
    "router_light":  "mistral:7b-instruct-v0.3-q5_0",
    "long_context":  "mistral-large:latest",
    "rag_worker":    "mistral:7b-instruct-v0.3-q5_0",
}
```

## Troubleshooting

### Common Issues

#### Ollama Connection Problems

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama service (if not running)
# Windows: ollama serve
# macOS/Linux: systemctl start ollama
```

#### Model Loading Issues

- **Timeouts on first call**: Large models need warm-up time. Increase HTTP timeout in Streamlit sidebar (default: 3600s)
- **Model not found**: Ensure models are pulled: `ollama pull <model-name>`
- **Out of memory**: Use smaller quantized models (q4_0, q4_1 instead of fp16)

#### RAG System Issues

- **ChromaDB errors**: Delete `rag_db/` directory to reset vector store
- **Embedding failures**: Verify `nomic-embed-text` model is available
- **No search results**: Check if documents were successfully indexed

#### Agent/Tools Issues

- **Web tools fail**: Check internet connectivity and firewall settings
- **OCR errors**: Ensure vision model (`llava:34b`) is available
- **Tool timeouts**: Adjust timeouts in Streamlit sidebar

#### Installation Issues

- **Import errors**: Verify Python 3.10+ and all requirements installed
- **Version conflicts**: Use fresh virtual environment
- **Missing dependencies**: Some features require optional packages:
  ```bash
  # For PDF processing
  pip install pypdf pymupdf
  
  # For enhanced web scraping
  pip install beautifulsoup4
  
  # For better search
  pip install duckduckgo-search
  ```

#### LangChain Compatibility

- **Timeout parameter**: `app.py` handles both `request_timeout` and `timeout` for different LangChain versions
- **API changes**: Update to latest LangChain versions for best compatibility

### Performance Optimization

#### Model Selection

- Use smaller models (7B-13B) for faster responses
- Reserve large models (70B+) for complex reasoning tasks
- Consider quantized models (q4_0, q4_K_M) for memory efficiency

#### RAG Optimization

- Adjust chunk sizes based on your document types
- Use filename filtering for focused searches
- Implement query preprocessing for better retrieval

#### Streamlit Performance

- Use soft timeouts to prevent UI blocking
- Enable streaming for long responses
- Limit context window sizes for faster processing

### Environment-Specific Notes

#### Corporate/Restricted Networks

- Web tools may fail due to proxy/firewall restrictions
- Use local file tools and disable web search
- Configure proxy settings if needed

#### Resource-Constrained Systems

- Use smaller models (7B parameters or less)
- Reduce context windows and max token limits
- Disable streaming for better stability
- Consider using Ollama on a separate server

### Debug Mode

Enable verbose logging for troubleshooting:

1. Set `verbose=True` in agent configuration
2. Enable "Verbose tools logs" in Streamlit sidebar
3. Check agent execution logs in `logs/` directory
4. Use LangSmith tracing for detailed debugging

## Future Enhancements

### Planned Features

- **Enhanced OCR**: Integration with specialized OCR models for better accuracy
- **Multi-modal RAG**: Support for image and video content in vector store
- **Advanced Agents**: More sophisticated tool orchestration and planning
- **Model Optimization**: Automatic model selection based on query complexity
- **UI Improvements**: Enhanced chat interface with better visualization

### Contributing

Contributions are welcome! Please focus on:

- Additional tool implementations
- Enhanced document processing capabilities
- Model router improvements
- Performance optimizations
- Documentation updates

### License

This project is designed for educational and development purposes. Please ensure compliance with Ollama model licenses and usage terms.

## Support

For issues and questions:

1. Check this documentation first
2. Review the troubleshooting section
3. Examine log files in `logs/` directory
4. Test with simpler models if issues persist
5. Verify Ollama service status and model availability

---

**Router** - Intelligent LLM routing with RAG capabilities
Built with LangChain, Ollama, ChromaDB, and Streamlit