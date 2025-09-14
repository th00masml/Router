from __future__ import annotations

import os
import argparse
from typing import List, Dict, Any

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import Client

from rag.store import similarity_search


def _ensure_langsmith(project: str | None, enable: bool, tags: List[str]) -> None:
    if enable:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        if project:
            os.environ["LANGCHAIN_PROJECT"] = project
    else:
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
    # Optionally validate API key presence
    if enable and not os.getenv("LANGSMITH_API_KEY"):
        raise RuntimeError("LANGSMITH_API_KEY is required when tracing is enabled.")


def build_llm(model: str, base_url: str, temperature: float = 0.0) -> ChatOllama:
    try:
        return ChatOllama(model=model, base_url=base_url, temperature=temperature, request_timeout=3600)
    except TypeError:
        return ChatOllama(model=model, base_url=base_url, temperature=temperature, timeout=3600)


def run_chat(model: str, base_url: str, prompt: str, sys: str, tags: List[str]) -> str:
    llm = build_llm(model, base_url)
    cfg = {"tags": ["router-app", "cli_chat", model] + tags, "metadata": {"entry": "chat"}}
    msgs = []
    if sys:
        msgs.append(SystemMessage(content=sys))
    msgs.append(HumanMessage(content=prompt))
    resp = llm.invoke(msgs, config=cfg)
    return resp.content


def run_rag(model: str, base_url: str, question: str, top_k: int, tags: List[str]) -> Dict[str, Any]:
    # Retrieval step (logs will appear for the LLM call; retrieval context captured in metadata on answer)
    hits = similarity_search(base_url, question, k=top_k)
    context_blocks = []
    files = []
    for i, h in enumerate(hits, 1):
        meta = h.get("metadata", {})
        fn = meta.get("filename", "")
        files.append(fn)
        context_blocks.append(f"[Doc {i} {fn}]\n{h['page_content']}")
    context = "\n\n".join(context_blocks)

    sys = (
        "You are a grounded assistant. Use ONLY the provided context to answer. "
        "If the answer is not in the context, say you don't have enough information. "
        "Cite doc numbers like [Doc 2] when referencing."
    )
    llm = build_llm(model, base_url)
    cfg = {
        "tags": ["router-app", "cli_rag", model] + tags,
        "metadata": {"k": top_k, "hits": len(hits), "files": files},
        "run_name": "cli_rag_answer",
    }
    out = llm.invoke([SystemMessage(content=sys), HumanMessage(content=f"Question: {question}\n\nContext:\n{context}")], config=cfg)
    return {"answer": out.content, "hits": hits}


def main() -> None:
    ap = argparse.ArgumentParser(description="LangSmith observability runner for the Router app")
    ap.add_argument("mode", nargs="?", choices=["chat", "rag"], default="chat", help="Workflow to run (default: chat)")
    ap.add_argument("--project", dest="project", default=os.getenv("LANGCHAIN_PROJECT", "Router-App"))
    ap.add_argument("--tags", dest="tags", default="", help="Comma-separated run tags")
    ap.add_argument("--enable", dest="enable", action="store_true", help="Enable LangSmith tracing")
    ap.add_argument("--base-url", dest="base_url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ap.add_argument("--model", dest="model", default="mistral:7b-instruct-v0.3-q5_0", help="Ollama model name")

    # chat arguments
    ap.add_argument("--prompt", dest="prompt", default="Hello, provide a concise introduction.")
    ap.add_argument("--system", dest="system", default="You are a helpful assistant.")

    # rag arguments
    ap.add_argument("--question", dest="question", default="What does the uploaded document say about risks?")
    ap.add_argument("--k", dest="k", type=int, default=5)

    args = ap.parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    _ensure_langsmith(args.project, args.enable, tags)

    # Ensure client is reachable when enabled (optional warm check)
    if args.enable:
        Client()

    if args.mode == "chat":
        out = run_chat(args.model, args.base_url, args.prompt, args.system, tags)
        print("=== Chat Response ===\n" + out)
    else:
        res = run_rag(args.model, args.base_url, args.question, args.k, tags)
        print("=== RAG Answer ===\n" + res["answer"])
        print("=== Retrieved Passages (truncated) ===")
        for i, h in enumerate(res["hits"], 1):
            fn = (h.get("metadata") or {}).get("filename", "")
            print(f"Doc {i} — {fn}\n" + h["page_content"][:300].replace("\n", " ") + "\n")


if __name__ == "__main__":
    main()
