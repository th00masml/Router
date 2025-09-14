import os
import base64
import json
import re
import requests
import yaml
import time
from typing import List, TypedDict, Dict, Any, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from agents.tools_agent import build_tools_agent, invoke_tools_agent
from tools.basic_tools import select_tools_for_query
from tools.basic_tools import set_timeouts
from rag.store import add_documents, similarity_search, delete_by_filename, reset_vectorstore, list_filenames, similarity_search_filtered
try:
    # For manual tracing of non-LangChain calls (e.g., OCR pipelines)
    from langsmith.run_helpers import traceable  # type: ignore
except ImportError:  # graceful fallback when LangSmith is not installed
    def traceable(func=None, **_kwargs):  # type: ignore
        """No-op decorator compatible with both @traceable and @traceable(...)."""
        if func is None:
            def _decorator(f):
                return f
            return _decorator
        return func

# -------------------------
# Widget key helper
# -------------------------
WKEY = "main"
def k(name: str) -> str:
    return f"{WKEY}:{name}"

# =========================
# 1) CONFIG: Ollama models
# =========================
MODELS: Dict[str, str] = {
    "chat":          "denisavetisyan/gemma3-27b-q4_k_m-32k:latest",
    "reasoning":     "deepseek-v2.5",
    "code":          "qwen3-coder:latest",
    "router_light":  "llama3.1:8b-instruct-q4_0",
    "long_context":  "mistral-small3.2:24b",
    "rag_worker":    "mistral:7b-instruct-v0.3-q5_0",
}
DEFAULT_MODE = "auto"  # "auto" routes by intent
SPECIAL_MODES: List[str] = []

# Per-mode soft budgets (seconds) AFTER the first token
DEFAULT_SOFT_TIMEOUTS: Dict[str, int] = {
    "chat":          300,
    "reasoning":     900,
    "code":          420,
    "router_light":  180,
    "long_context":  1200,
    "rag_worker":    300,
}

# ===================================
# 2) Heuristic intent router
# ===================================
def route_intent(user_text: str) -> str:
    t = user_text.lower()
    code_kw = ("code", "python", "typescript", "refactor", "test", "bug", "compile", "function", "class", "api")
    if any(k in t for k in code_kw): return "code"
    long_kw = ("pdf", "document", "long text", "analyze file", "sections", "table of contents")
    if any(k in t for k in long_kw) or len(t.split()) > 800: return "long_context"
    reason_kw = ("prove", "step by step", "derivation", "logic", "reasoning", "mathematics", "proof")
    if any(k in t for k in reason_kw): return "reasoning"
    rag_kw = ("extract", "summarize", "bullet points", "entities", "regex", "parse", "rag")
    if any(k in t for k in rag_kw): return "rag_worker"
    lite_kw = ("short", "tl;dr", "brief", "concise")
    if any(k in t for k in lite_kw): return "router_light"
    return "chat"

# ==========================
# 3) Streamlit init
# ==========================
st.set_page_config(page_title="Router Chat - LangGraph + Ollama", layout="wide")

def init_session():
    if "history" not in st.session_state:
        st.session_state.history: List[AnyMessage] = []
    if "mode" not in st.session_state:
        st.session_state.mode: str = DEFAULT_MODE
    if "route_used" not in st.session_state:
        st.session_state.route_used: Optional[str] = None
    if "llm_map" not in st.session_state:
        st.session_state.llm_map = None
    if "stop_flag" not in st.session_state:
        st.session_state.stop_flag = False
    if "use_streaming" not in st.session_state:
        st.session_state.use_streaming = True
    if "cfg_base_url" not in st.session_state:
        st.session_state.cfg_base_url = None
    if "cfg_temp" not in st.session_state:
        st.session_state.cfg_temp = None
    if "cfg_request_timeout" not in st.session_state:
        st.session_state.cfg_request_timeout = None
    if "use_tools" not in st.session_state:
        st.session_state.use_tools = False
    if "tools_exec_map" not in st.session_state:
        st.session_state.tools_exec_map = {}
    if "pending_input" not in st.session_state:
        st.session_state.pending_input = None
    if "cfg_ls_enabled" not in st.session_state:
        st.session_state.cfg_ls_enabled = bool(os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("1","true","yes"))
    if "cfg_ls_project" not in st.session_state:
        st.session_state.cfg_ls_project = os.getenv("LANGCHAIN_PROJECT", "Router-App")
    if "cfg_ls_tags" not in st.session_state:
        st.session_state.cfg_ls_tags = ""

init_session()

# ==========================
# Run config builder (LangSmith)
# ==========================
# (Removed duplicate definition of _build_run_config)

# ==========================
# 4) Sidebar (settings) â€” all widgets have unique keys
# ==========================
with st.sidebar:
    st.header("Settings")

    base_url = st.text_input(
        "Ollama base_url (optional)",
        value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        key=k("base_url"),
    )
    # Quick Ollama status
    try:
        _installed = _ollama_list_models(base_url)
        if not _installed:
            st.caption("Ollama not reachable or no models installed at this base_url.")
        else:
            sample = ", ".join(sorted(list(_installed))[:8])
            more = " ..." if len(_installed) > 8 else ""
            st.caption(f"Installed models: {sample}{more}")
            _missing = [m for m in MODELS.values() if m not in _installed]
            if _missing:
                preview = ", ".join(_missing[:4]) + (" ..." if len(_missing) > 4 else "")
                st.warning(f"Missing configured models: {preview}")
    except Exception:
        pass

    temperature = st.slider(
        "Temperature",
        0.0, 1.0, 0.0, 0.1,
        key=k("temperature"),
    )

    # HTTP client timeout to Ollama. Increase for long generations and cold starts.
    request_timeout = st.number_input(
        "HTTP request timeout to Ollama (seconds)",
        min_value=30, max_value=86_400, value=3600, step=30,
        help="Client-side HTTP timeout in ChatOllama. Make this large to survive model warm-up/pulls.",
        key=k("request_timeout"),
    )

    sys_prompt = st.text_area(
        "System prompt (optional)",
        value="You are a helpful assistant.",
        height=80,
        key=k("sys_prompt"),
    )

    def _mode_label(m: str) -> str:
        if m == "auto":
            return "Auto (router)"
        return m

    mode = st.selectbox(
        "Mode",
        options=["auto"] + SPECIAL_MODES + list(MODELS.keys()),
        format_func=_mode_label,
        key=k("mode"),
    )
    if mode != st.session_state.mode:
        st.session_state.mode = mode

    st.checkbox(
        "Streaming",
        value=st.session_state.get("use_streaming", True),
        key=k("use_streaming"),
        help="Stream tokens so you can stop generation.",
    )
    st.session_state.use_streaming = st.session_state[k("use_streaming")]

    st.checkbox(
        "Enable tools / web access",
        value=st.session_state.get("use_tools", False),
        key=k("use_tools"),
        help="Allow the assistant to use web search/fetch, calculator, and file tools for this chat.",
    )
    st.session_state.use_tools = st.session_state[k("use_tools")]

    # Tools settings (visible when enabled)
    tools_alias = st.session_state.get("cfg_tools_alias", "router_light")
    tools_iterations = st.session_state.get("cfg_tools_iterations", 10)
    tools_verbose = st.session_state.get("cfg_tools_verbose", False)
    tools_strict = st.session_state.get("cfg_tools_strict", False)
    if st.session_state.use_tools:
        tools_alias = st.selectbox(
            "Tools model",
            options=list(MODELS.keys()),
            index=list(MODELS.keys()).index(tools_alias) if tools_alias in MODELS else 0,
            key=k("tools_alias"),
            help="Model used internally by the tools agent (niezaleÅ¼ny od trybu wybranego powyÅ¼ej)",
        )
        tools_iterations = st.slider(
            "Max tool iterations",
            min_value=3, max_value=30, value=tools_iterations, step=1,
            key=k("tools_iterations"),
        )
        tools_verbose = st.checkbox(
            "Verbose tools logs",
            value=tools_verbose,
            key=k("tools_verbose"),
        )
        tools_strict = st.checkbox(
            "Strict tools mode (fast finish)",
            value=st.session_state.get("cfg_tools_strict", False),
            key=k("tools_strict"),
            help="For date/time queries force 1-step (now/today) then Final Answer; in general prefer â‰¤2 steps.",
        )
        web_get_timeout = st.number_input(
            "Web fetch timeout (s)",
            min_value=3, max_value=120, value=st.session_state.get("cfg_web_get_timeout", 15), step=1,
            key=k("web_get_timeout"),
        )
        web_search_timeout = st.number_input(
            "Web search timeout (s)",
            min_value=3, max_value=60, value=st.session_state.get("cfg_web_search_timeout", 8), step=1,
            key=k("web_search_timeout"),
        )
        save_steps = st.checkbox(
            "Save agent steps to JSON (logs/)",
            value=st.session_state.get("cfg_tools_save_logs", True),
            key=k("tools_save_logs"),
        )
    # If any tools setting changed, clear cached executors
    if (
        tools_alias != st.session_state.get("cfg_tools_alias") or
        tools_iterations != st.session_state.get("cfg_tools_iterations") or
        tools_verbose != st.session_state.get("cfg_tools_verbose") or
        tools_strict != st.session_state.get("cfg_tools_strict") or
        (st.session_state.use_tools and (
            st.session_state[k("web_get_timeout")] != st.session_state.get("cfg_web_get_timeout") or
            st.session_state[k("web_search_timeout")] != st.session_state.get("cfg_web_search_timeout")
        ))
    ):
        st.session_state.tools_exec_map = {}
        st.session_state.cfg_tools_alias = tools_alias
        st.session_state.cfg_tools_iterations = tools_iterations
        st.session_state.cfg_tools_verbose = tools_verbose
        st.session_state.cfg_tools_strict = tools_strict
        if st.session_state.use_tools:
            st.session_state.cfg_web_get_timeout = st.session_state[k("web_get_timeout")]
            st.session_state.cfg_web_search_timeout = st.session_state[k("web_search_timeout")]
            # Apply to tools module
            set_timeouts(st.session_state.cfg_web_get_timeout, st.session_state.cfg_web_search_timeout)
    # Persist save_logs switch
    if st.session_state.use_tools:
        st.session_state.cfg_tools_save_logs = st.session_state[k("tools_save_logs")]

    st.markdown("Soft timeout (frontend, starts AFTER first token)")
    use_soft_timeout = st.checkbox(
        "Enable soft timeout",
        value=True,
        help="If enabled, the stream loop will stop after the budget below, counting from the first token.",
        key=k("use_soft_timeout"),
    )

    default_for_mode = DEFAULT_SOFT_TIMEOUTS.get(
        st.session_state.mode if st.session_state.mode != "auto" else "chat",
        300,
    )
    soft_timeout_seconds = st.number_input(
        "Soft timeout seconds (after first token)",
        min_value=30, max_value=86_400, value=default_for_mode, step=30,
        help="Applies only after the first token arrives.",
        key=k("soft_timeout_seconds"),
    )

    # (OCR model moved into the OCR expander to avoid duplication)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear history", key=k("clear_history_btn")):
            st.session_state.history = []
            st.session_state.route_used = None
            st.rerun()
    with col2:
        if st.button("Stop generation", key=k("stop_generation_btn")):
            st.session_state.stop_flag = True

    st.markdown("---")
    st.caption("Tip: cold starts can take minutes. Use a large HTTP timeout and rely on the stop button if needed.")

    with st.expander("LangSmith Observability", expanded=False):
        st.caption("Requires LANGSMITH_API_KEY in your environment. Project and tags are optional.")
        ls_enabled = st.checkbox(
            "Enable LangSmith tracing",
            value=st.session_state.get("cfg_ls_enabled", False),
            key=k("ls_enabled"),
        )
        ls_project = st.text_input(
            "Project",
            value=st.session_state.get("cfg_ls_project", "Router-App"),
            key=k("ls_project"),
        )
        ls_tags = st.text_input(
            "Run tags (comma separated)",
            value=st.session_state.get("cfg_ls_tags", ""),
            key=k("ls_tags"),
        )
        # Apply env immediately if changed
        if ls_enabled != st.session_state.cfg_ls_enabled or ls_project != st.session_state.cfg_ls_project:
            st.session_state.cfg_ls_enabled = ls_enabled
            st.session_state.cfg_ls_project = ls_project
            os.environ["LANGCHAIN_PROJECT"] = ls_project
            if ls_enabled:
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                # Default LangSmith endpoint if not set
                os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
            else:
                os.environ.pop("LANGCHAIN_TRACING_V2", None)
        if ls_tags != st.session_state.cfg_ls_tags:
            st.session_state.cfg_ls_tags = ls_tags

# ==========================
# 5) LLM map (with request_timeout)
# ==========================
def build_llms(base_url: str, temperature: float, request_timeout: int) -> Dict[str, ChatOllama]:
    # Some versions of langchain-ollama use `timeout` instead of `request_timeout`.
    # Try `request_timeout`, and if it fails, fall back to `timeout`.
    def make_client(model: str) -> ChatOllama:
        try:
            return ChatOllama(
                model=model,
                temperature=temperature,
                base_url=base_url,
                request_timeout=request_timeout,
            )
        except TypeError:
            return ChatOllama(
                model=model,
                temperature=temperature,
                base_url=base_url,
                timeout=request_timeout,
            )

    return {alias: make_client(name) for alias, name in MODELS.items()}

# --------------------------
# Ollama connectivity helpers
# --------------------------
def _ollama_list_models(base_url: str) -> set[str]:
    try:
        base = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        resp = requests.get(f"{base}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json() or {}
        models = set()
        for m in data.get("models", []):
            name = m.get("name") or m.get("model") or ""
            if name:
                models.add(name)
        return models
    except Exception:
        return set()

models_fingerprint = json.dumps(MODELS, sort_keys=True)
needs_rebuild = (
    st.session_state.llm_map is None
    or st.session_state.cfg_base_url != base_url
    or st.session_state.cfg_temp != temperature
    or st.session_state.cfg_request_timeout != request_timeout
    or st.session_state.get("cfg_models_fingerprint") != models_fingerprint
)
if needs_rebuild:
    st.session_state.llm_map = build_llms(base_url, temperature, request_timeout)
    st.session_state.cfg_base_url = base_url
    st.session_state.cfg_temp = temperature
    st.session_state.cfg_request_timeout = request_timeout
    st.session_state.cfg_models_fingerprint = models_fingerprint
    # Reset tools executors cache because LLM clients changed
    st.session_state.tools_exec_map = {}

# ==========================
# 6) Helpers
# ==========================
def ensure_system_prompt(history: List[AnyMessage], sys_prompt: str) -> List[AnyMessage]:
    if sys_prompt and (not history or not isinstance(history[0], SystemMessage)):
        return [SystemMessage(content=sys_prompt)] + history
    return history

def resolve_route(eff_history: List[AnyMessage], chosen_mode: str) -> str:
    if chosen_mode != "auto":
        return chosen_mode
    last_user = ""
    for m in reversed(eff_history):
        if isinstance(m, HumanMessage):
            last_user = m.content
            break
    return route_intent(last_user)

# ==========================
# 7) Header and status
# ==========================
st.title("Router Chat - LangGraph + Ollama")
status_cols = st.columns(4)
with status_cols[0]:
    st.metric("Mode", "Auto" if st.session_state.mode == "auto" else st.session_state.mode)
with status_cols[1]:
    st.metric("Temperature", f"{temperature:.1f}")
with status_cols[2]:
    st.metric("Last route", st.session_state.route_used or "-")
with status_cols[3]:
    st.metric("HTTP timeout (s)", request_timeout)

st.divider()

# ==========================
# 8) Quick actions: Summarize URL (non-agent)
# ==========================
with st.expander("Quick: Summarize URL", expanded=False):
    url_in = st.text_input("Page URL (http/https)", key=k("qa_url"))
    if st.button("Summarize", key=k("qa_summarize_btn")):
        if not url_in or not re.match(r"^https?://", url_in.strip()):
            st.error("Please provide a valid http(s) URL.")
        else:
            try:
                with st.spinner("Fetching page..."):
                    resp = requests.get(url_in.strip(), headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
                    resp.raise_for_status()
                    content_type = resp.headers.get("Content-Type", "")
                    text = resp.text
                # Basic cleanup for HTML
                if "html" in content_type.lower():
                    try:
                        from bs4 import BeautifulSoup  # type: ignore
                        soup = BeautifulSoup(text, "html.parser")
                        for tag in soup(["script", "style", "noscript"]):
                            tag.decompose()
                        text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()
                    except Exception:
                        text = re.sub(r"<[^>]+>", " ", text)
                        text = re.sub(r"\s+", " ", text).strip()
                # Truncate to keep prompt reasonable
                if len(text) > 10000:
                    text = text[:10000] + "\n...[truncated]"

                # Pick a concise model
                alias = "router_light" if "router_light" in st.session_state.llm_map else "chat"
                llm = st.session_state.llm_map.get(alias)
                if llm is None:
                    st.error("No LLM available. Check model settings.")
                else:
                    sys_msg = (
                        "You are a concise web page summarizer. "
                        "Produce 5-8 bullet points highlighting key facts and takeaways. "
                        "Use neutral tone. Include the source URL at the end."
                    )
                    messages = [
                        SystemMessage(content=sys_msg),
                        HumanMessage(content=f"URL: {url_in}\n\nContent to summarize:\n{text}"),
                    ]
                    with st.spinner("Summarizing..."):
                        out = llm.invoke(
                            messages,
                            config=_build_run_config(
                                "quick_summarize",
                                name="quick_summarize",
                                meta={
                                    "url": url_in,
                                    "content_type": content_type,
                                    "content_len": len(text),
                                },
                            ),
                        )
                    st.markdown(out.content)
                    st.caption(f"Source: {url_in}")
            except Exception as e:
                st.error(f"Summarization failed: {e}")

with st.expander("Document OCR / Reader", expanded=False):
    st.caption("Upload an image or PDF, or provide a URL. PDFs with selectable text will be read directly; scanned PDFs are OCRâ€™d per page if PyMuPDF is available.")

    # Single place to configure OCR model (with presets)
    current_ocr = st.session_state.get("cfg_ocr_model") or os.getenv("OLLAMA_OCR_MODEL", "llava:34b")
    preset_models = ["llava:34b", "(Custom...)"]
    if current_ocr not in preset_models:
        preset_models = [current_ocr] + [m for m in preset_models if m != current_ocr]
    sel = st.selectbox(
        "OCR model (Ollama)",
        options=preset_models,
        index=0,
        key=k("ocr_any_model_sel"),
        help="Select a local vision model available in Ollama. Use Custom to type any name.",
    )
    if sel == "(Custom...)":
        custom = st.text_input(
            "Custom OCR model name",
            value=current_ocr if current_ocr not in ["llava:34b"] else "",
            key=k("ocr_any_model_custom"),
        )
        chosen_model = (custom or current_ocr).strip()
    else:
        chosen_model = sel
    if chosen_model and chosen_model != st.session_state.get("cfg_ocr_model"):
        st.session_state.cfg_ocr_model = chosen_model
        # Keep env + tools module in sync so tools.ocr_image uses the same model
        os.environ["OLLAMA_OCR_MODEL"] = chosen_model
        try:
            import tools.basic_tools as _bt  # type: ignore
            _bt.OLLAMA_OCR_MODEL = chosen_model  # type: ignore[attr-defined]
        except Exception:
            pass

    # Prompt configuration (default or custom)
    DEFAULT_OCR_PROMPT = (
        "You are a highâ€‘accuracy OCR transcription engine.\n"
        "Goal: transcribe ALL visible text from the provided page image(s) ASâ€‘IS.\n"
        "Rules:\n"
        "- Preserve reading order and line breaks. Do not reflow into paragraphs.\n"
        "- Keep punctuation, capitalization, diacritics and spacing exactly as written.\n"
        "- Read both printed and handwritten text, including stamps and annotations.\n"
        "- For checkboxes, output '[x]' (checked) or '[ ]' (unchecked) before the label when visually clear.\n"
        "- Include form labels AND filled values on the same line as 'Label: Value' when obvious.\n"
        "- For tables, keep one row per line using pipes, e.g. '| col1 | col2 |'. Keep cell text verbatim.\n"
        "- If a word/field is illegible, write '[?]' in place; do NOT guess.\n"
        "- Do NOT summarize, translate, interpret, or add commentary. Do NOT fabricate content.\n"
        "- For multiâ€‘page input, we will combine results per page outside of you.\n"
        "Output: plain text only."
    )
    current_prompt = st.session_state.get("cfg_ocr_prompt") or DEFAULT_OCR_PROMPT
    with st.expander("OCR Prompt (advanced)", expanded=False):
        use_custom_prompt = st.checkbox(
            "Use custom OCR prompt",
            value=bool(st.session_state.get("cfg_ocr_prompt_custom", False)),
            key=k("ocr_prompt_custom_flag"),
            help="Override the default OCR instruction with your own.",
        )
        if use_custom_prompt:
            custom_prompt = st.text_area(
                "OCR system prompt",
                value=current_prompt if current_prompt else DEFAULT_OCR_PROMPT,
                height=180,
                key=k("ocr_prompt_text"),
            )
            st.session_state.cfg_ocr_prompt_custom = True
            st.session_state.cfg_ocr_prompt = custom_prompt or DEFAULT_OCR_PROMPT
        else:
            st.session_state.cfg_ocr_prompt_custom = False
            st.session_state.cfg_ocr_prompt = DEFAULT_OCR_PROMPT

    doc_up = st.file_uploader(
        "Upload document (image or PDF)",
        type=["png", "jpg", "jpeg", "webp", "bmp", "pdf"],
        accept_multiple_files=False,
        key=k("ocr_any_upload"),
    )
    doc_url = st.text_input("Or URL (image/PDF)", key=k("ocr_any_url"))
    col_ocr1, col_ocr2, col_ocr3 = st.columns([1,1,1])
    with col_ocr1:
        max_pdf_pages = st.number_input("Max PDF pages to OCR", min_value=1, max_value=50, value=3, step=1, key=k("ocr_any_max_pages"))
    with col_ocr2:
        ocr_pdf_force = st.checkbox("Force OCR for PDFs", value=False, key=k("ocr_any_force"), help="Ignore embedded PDF text and OCR rendered pages. Helps capture handwriting and stamps.")
    with col_ocr3:
        ocr_pdf_dpi = st.slider("Render DPI", min_value=120, max_value=600, value=360, step=20, key=k("ocr_any_dpi"), help="Higher DPI improves OCR for handwriting but is slower.")
    # Handwriting enhancement controls
    col_hw1, col_hw2 = st.columns([1,2])
    with col_hw1:
        ocr_hw_enhance = st.checkbox("Enhance handwriting", value=False, key=k("ocr_hw_enhance"), help="Apply contrast + binarization to improve faint handwriting.")
    with col_hw2:
        ocr_hw_thresh = st.slider("Binarization threshold", min_value=100, max_value=220, value=170, step=5, key=k("ocr_hw_thresh"))
    col_tile1, col_tile2, col_tile3 = st.columns([1,1,1])
    with col_tile1:
        ocr_tile_enable = st.checkbox("Tile pages/images", value=False, key=k("ocr_tile_enable"), help="Split into smaller tiles before OCR to improve small handwriting recognition.")
    with col_tile2:
        ocr_tile_size = st.slider("Tile size (px)", min_value=512, max_value=2000, value=1024, step=64, key=k("ocr_tile_size"))
    with col_tile3:
        ocr_tile_overlap = st.slider("Tile overlap (px)", min_value=0, max_value=400, value=100, step=10, key=k("ocr_tile_overlap"))

    def _fetch_url_bytes(url: str, timeout_s: int) -> Tuple[bytes, str]:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout_s)
        r.raise_for_status()
        return r.content, r.headers.get("Content-Type", "")

    @traceable(name="pdf_extract_text", run_type="tool")
    def _pdf_extract_text(data: bytes) -> str:
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception:
            return ""
        try:
            import io as _io
            reader = PdfReader(_io.BytesIO(data))
            pages = [p.extract_text() or "" for p in reader.pages]
            out = "\n\n".join(pages)
            # Include AcroForm text fields if present
            try:
                tf = getattr(reader, "get_form_text_fields", None)
                if callable(tf):
                    fields = tf()
                    if fields:
                        lines = [f"{k}: {v}" for k, v in fields.items()]
                        out += "\n\n[PDF Form Fields]\n" + "\n".join(lines)
                else:
                    gf = getattr(reader, "get_fields", None)
                    if callable(gf):
                        fields = gf() or {}
                        lines = []
                        for k, v in fields.items():
                            val = None
                            if isinstance(v, dict):
                                val = v.get("/V") or v.get("V")
                            if val is None:
                                val = getattr(v, "value", None)
                            lines.append(f"{k}: {val if val is not None else ''}")
                        if lines:
                            out += "\n\n[PDF Form Fields]\n" + "\n".join(lines)
            except Exception:
                pass
            return out
        except Exception:
            return ""

    def _preprocess_image_bytes(raw_png: bytes, threshold: int, enable: bool, gamma: float = 1.0) -> bytes:
        if not enable:
            return raw_png
        try:
            from PIL import Image, ImageOps, ImageFilter  # type: ignore
            import io as _io
            im = Image.open(_io.BytesIO(raw_png)).convert("L")
            im = ImageOps.autocontrast(im)
            im = im.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
            # Optional gamma correction
            try:
                g = float(gamma)
                if abs(g - 1.0) > 1e-3:
                    im = ImageOps.gamma(im, g)
            except Exception:
                pass
            # Simple global threshold
            thr = int(threshold)
            im = im.point(lambda p: 255 if p >= thr else 0)
            buf = _io.BytesIO()
            im.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            return raw_png

    def _tile_image_to_b64(raw_png: bytes, size: int, overlap: int, threshold: int, enhance: bool, gamma: float = 1.0) -> list[str]:
        try:
            from PIL import Image  # type: ignore
            import io as _io
            img = Image.open(_io.BytesIO(raw_png)).convert("L")
            W, H = img.size
            step = max(1, int(size) - int(overlap))
            out: list[str] = []
            for y in range(0, H, step):
                for x in range(0, W, step):
                    crop = img.crop((x, y, min(x + size, W), min(y + size, H)))
                    buf = _io.BytesIO()
                    crop.save(buf, format="PNG")
                    tile_bytes = _preprocess_image_bytes(buf.getvalue(), int(threshold), bool(enhance), float(gamma))
                    out.append(base64.b64encode(tile_bytes).decode())
            return out
        except Exception:
            # fallback: single b64
            return [base64.b64encode(raw_png).decode()]

    def _pdf_pages_to_images_b64(data: bytes, max_pages: int, dpi: int = 300) -> list[str]:
        try:
            import fitz  # PyMuPDF
        except Exception:
            return []
        imgs: list[str] = []
        try:
            import io as _io
            doc = fitz.open(stream=data, filetype="pdf")
            n = min(len(doc), max_pages)
            for i in range(n):
                page = doc.load_page(i)
                # Render at chosen DPI for better OCR of handwriting
                pix = page.get_pixmap(dpi=int(dpi), alpha=False)
                png = pix.tobytes("png")
                if bool(ocr_tile_enable):
                    # Tile the page image for better small-text capture
                    tiles = _tile_image_to_b64(
                        png,
                        int(ocr_tile_size),
                        int(ocr_tile_overlap),
                        int(ocr_hw_thresh),
                        bool(ocr_hw_enhance),
                    )
                    imgs.extend(tiles)
                else:
                    png = _preprocess_image_bytes(png, int(ocr_hw_thresh), bool(ocr_hw_enhance))
                    imgs.append(base64.b64encode(png).decode())
            return imgs
        except Exception:
            return []

    def _llava_ocr_images(b64_list: list[str], model: str, base_url: str, timeout_s: int, prompt: str) -> list[str]:
        out: list[str] = []
        base = base_url.rstrip('/')
        for b64 in b64_list:
            ok = False
            text_piece = ""
            # First try /api/generate (works for most multimodal models like LLaVA)
            try:
                resp = requests.post(
                    f"{base}/api/generate",
                    json={"model": model, "prompt": prompt, "images": [b64], "stream": False},
                    timeout=timeout_s,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    text_piece = data.get("response", "")
                    ok = True
                else:
                    # fall through to chat fallback
                    ok = False
            except Exception:
                ok = False

            if not ok or not text_piece:
                # Fallback to /api/chat (some builds/models expect chat API for images)
                try:
                    resp2 = requests.post(
                        f"{base}/api/chat",
                        json={
                            "model": model,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": prompt,
                                    "images": [b64],
                                }
                            ],
                            "stream": False,
                        },
                        timeout=timeout_s,
                    )
                    resp2.raise_for_status()
                    data2 = resp2.json()
                    # chat schema: { message: { role, content }, ... }
                    text_piece = (
                        (data2.get("message") or {}).get("content")
                        or data2.get("response", "")
                    )
                    ok = True
                except Exception as e:
                    # Provide a clearer error for non-vision models
                    raise RuntimeError(
                        f"Vision OCR failed for model '{model}'. Ensure the selected model supports images (e.g., llava:34b). "
                        f"Original error: {e}"
                    )

            out.append(text_piece or "")
        return out

    @traceable(name="ocr_document", run_type="chain")
    def _trace_ocr(meta: Dict[str, Any], result_preview: str) -> str:
        return result_preview

    if st.button("Run OCR / Read", key=k("ocr_any_run")):
        if not doc_up and not doc_url:
            st.error("Please upload a document or provide a URL.")
        else:
            try:
                # Acquire bytes and type
                raw: bytes
                ctype = ""
                fname = ""
                if doc_up:
                    raw = doc_up.read()
                    fname = doc_up.name or ""
                else:
                    url = doc_url.strip()
                    if not re.match(r"^https?://", url):
                        st.error("Provide a valid http(s) URL.")
                        raise RuntimeError("invalid url")
                    timeout_get = int(st.session_state.get("cfg_web_get_timeout", 15))
                    raw, ctype = _fetch_url_bytes(url, timeout_get)
                    fname = url
                if not raw:
                    st.error("Empty document content.")
                    raise RuntimeError("empty content")

                # Decide path based on type/extension
                is_pdf = (
                    (fname.lower().endswith(".pdf")) or ("pdf" in ctype.lower())
                )
                base_url_eff = st.session_state.cfg_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                model_eff = st.session_state.get("cfg_ocr_model") or os.getenv("OLLAMA_OCR_MODEL", "llava:34b")
                timeout_eff = int(os.getenv("OLLAMA_TIMEOUT", "180"))

                if is_pdf:
                    # 1) Try direct text extraction first (incl. form fields) unless forced OCR
                    direct_text = _pdf_extract_text(raw) if not ocr_pdf_force else ""
                    if direct_text.strip():
                        st.text_area("Extracted text (PDF)", direct_text, height=260)
                        st.caption("Source: direct PDF text extraction")
                        # Trace a compact summary
                        try:
                            _trace_ocr(
                                {
                                    "source": "pdf_text",
                                    "filename": fname,
                                    "bytes": len(raw),
                                    "model": model_eff,
                                    "base_url": base_url_eff,
                                    "force_ocr": bool(ocr_pdf_force),
                                },
                                direct_text[:500],
                            )
                        except Exception:
                            pass
                    else:
                        # 2) Render pages and OCR via LLaVA
                        b64_pages = _pdf_pages_to_images_b64(raw, int(max_pdf_pages), int(ocr_pdf_dpi))
                        if not b64_pages:
                            st.error("No text in PDF and cannot render pages for OCR. Install 'pymupdf' to enable scanned PDF OCR.")
                        else:
                            with st.spinner("OCRâ€™ing PDF pages..."):
                                texts = _llava_ocr_images(
                                    b64_pages,
                                    model_eff,
                                    base_url_eff,
                                    timeout_eff,
                                    st.session_state.get("cfg_ocr_prompt") or DEFAULT_OCR_PROMPT,
                                )
                            merged = []
                            for i, t in enumerate(texts, 1):
                                merged.append(f"[Page {i}]\n{t}")
                            out_txt = "\n\n".join(merged)
                            st.text_area("Extracted text (OCR)", out_txt, height=300)
                            st.caption(f"Model: {model_eff} â€¢ Pages OCRâ€™d: {len(texts)}")
                            try:
                                _trace_ocr(
                                    {
                                        "source": "pdf_ocr",
                                        "filename": fname,
                                        "bytes": len(raw),
                                        "model": model_eff,
                                        "base_url": base_url_eff,
                                        "pages": len(texts),
                                        "dpi": int(ocr_pdf_dpi),
                                    },
                                    out_txt[:500],
                                )
                            except Exception:
                                pass
                else:
                    # Try image OCR or plain-text decode for text formats
                    # Heuristic: if looks like text, show directly
                    try:
                        text_guess = raw.decode("utf-8")
                        looks_text = len(re.sub(r"\s+", "", text_guess)) > 0
                        if looks_text:
                            # If HTML, strip tags
                            if "<" in text_guess and ">" in text_guess:
                                try:
                                    from bs4 import BeautifulSoup  # type: ignore
                                    soup = BeautifulSoup(text_guess, "html.parser")
                                    for tag in soup(["script", "style", "noscript"]):
                                        tag.decompose()
                                    text_guess = re.sub(r"\s+", " ", soup.get_text(" ")).strip()
                                except Exception:
                                    text_guess = re.sub(r"<[^>]+>", " ", text_guess)
                                    text_guess = re.sub(r"\s+", " ", text_guess).strip()
                            st.text_area("Document text", text_guess[:20000] + ("\n...[truncated]" if len(text_guess) > 20000 else ""), height=260)
                            st.caption("Source: direct text content")
                            try:
                                _trace_ocr(
                                    {
                                        "source": "text_direct",
                                        "filename": fname,
                                        "bytes": len(raw),
                                    },
                                    text_guess[:500],
                                )
                            except Exception:
                                pass
                        else:
                            raise UnicodeDecodeError("utf-8", b"", 0, 1, "not text")
                    except Exception:
                        # Treat as image (apply handwriting enhancement/tiling if enabled)
                        if bool(ocr_tile_enable):
                            # Tile the input image
                            tiles_b64 = _tile_image_to_b64(
                                raw,
                                int(ocr_tile_size),
                                int(ocr_tile_overlap),
                                int(ocr_hw_thresh),
                                bool(ocr_hw_enhance),
                            )
                            with st.spinner("Running OCR on tiles..."):
                                texts = _llava_ocr_images(
                                    tiles_b64,
                                    model_eff,
                                    base_url_eff,
                                    timeout_eff,
                                    st.session_state.get("cfg_ocr_prompt") or DEFAULT_OCR_PROMPT,
                                )
                            text = "\n\n".join(t for t in texts if t)
                        else:
                            raw_img = _preprocess_image_bytes(raw, int(ocr_hw_thresh), bool(ocr_hw_enhance))
                            b64 = base64.b64encode(raw_img).decode()
                            with st.spinner("Running OCR..."):
                                texts = _llava_ocr_images(
                                    [b64],
                                    model_eff,
                                    base_url_eff,
                                    timeout_eff,
                                    st.session_state.get("cfg_ocr_prompt") or DEFAULT_OCR_PROMPT,
                                )
                            text = texts[0] if texts else ""
                        if not text:
                            st.warning("OCR returned no text.")
                        else:
                            st.text_area("Extracted text", text, height=220)
                            st.caption(f"Model: {model_eff}")
                            try:
                                _trace_ocr(
                                    {
                                        "source": "image_ocr",
                                        "filename": fname,
                                        "bytes": len(raw),
                                        "model": model_eff,
                                        "base_url": base_url_eff,
                                    },
                                    text[:500],
                                )
                            except Exception:
                                pass
                
            except Exception as e:
                st.error(f"OCR/Read failed: {e}")

# ==========================
# 9) RAG: Documents and Ask
# ==========================
with st.expander("RAG: Documents", expanded=False):
    st.caption("Upload documents to a local vector DB (Chroma). They will be chunked and embedded via Ollama embeddings (nomic-embed-text).")
    up = st.file_uploader("Upload files (txt, md, html, json, pdf)", type=["txt", "md", "html", "json", "pdf"], accept_multiple_files=True, key=k("rag_upload"))
    if up:
        base_url_eff = st.session_state.cfg_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        total_chunks = 0
        import io, json as _json, re as _re
        try:
            from pypdf import PdfReader as _PdfReader
        except Exception:
            _PdfReader = None  # type: ignore
        for f in up:
            name = f.name
            data = f.read()
            text = ""
            try:
                if name.lower().endswith((".txt", ".md", ".html")):
                    text = data.decode("utf-8", errors="ignore")
                    if name.lower().endswith(".html"):
                        try:
                            from bs4 import BeautifulSoup  # type: ignore
                            soup = BeautifulSoup(text, "html.parser")
                            for tag in soup(["script", "style", "noscript"]):
                                tag.decompose()
                            text = _re.sub(r"\s+", " ", soup.get_text(" ")).strip()
                        except Exception:
                            text = _re.sub(r"<[^>]+>", " ", text)
                elif name.lower().endswith(".json"):
                    text = data.decode("utf-8", errors="ignore")
                elif name.lower().endswith(".pdf"):
                    if _PdfReader is None:
                        st.warning("PDF support is not available (pypdf not installed). Install 'pypdf' to enable PDF text extraction.")
                        text = ""
                    else:
                        try:
                            reader = _PdfReader(io.BytesIO(data))
                            pages = [p.extract_text() or "" for p in reader.pages]
                            text = "\n\n".join(pages)
                            # Include AcroForm text fields if available
                            try:
                                tf = getattr(reader, "get_form_text_fields", None)
                                if callable(tf):
                                    fields = tf()
                                    if fields:
                                        lines = [f"{k}: {v}" for k, v in fields.items()]
                                        text += "\n\n[PDF Form Fields]\n" + "\n".join(lines)
                                else:
                                    gf = getattr(reader, "get_fields", None)
                                    if callable(gf):
                                        fields = gf() or {}
                                        lines = []
                                        for k, v in fields.items():
                                            val = None
                                            if isinstance(v, dict):
                                                val = v.get("/V") or v.get("V")
                                            if val is None:
                                                val = getattr(v, "value", None)
                                            lines.append(f"{k}: {val if val is not None else ''}")
                                        if lines:
                                            text += "\n\n[PDF Form Fields]\n" + "\n".join(lines)
                            except Exception:
                                pass
                            if not text.strip():
                                st.info(
                                    f"No extractable text detected in {name}. It may be a scanned/image-only PDF. "
                                    "Use the 'Document OCR / Reader' section above to OCR pages or run external OCR to convert it to text."
                                )
                        except Exception as e:
                            st.warning(f"Failed to read PDF {name}: {e}")
                            text = ""
            except Exception as e:
                st.warning(f"Failed to process {name}: {e}")
                text = ""
            if text:
                total_chunks += add_documents(base_url_eff, [{"text": text, "metadata": {"filename": name}}])
        st.success(f"Indexed ~{total_chunks} chunks from {len(up)} file(s).")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Clear RAG index (all docs)", key=k("rag_clear_all")):
            reset_vectorstore()
            st.success("RAG index cleared.")
    with col_b:
        del_name = st.text_input("Delete by filename (exact)", key=k("rag_del_name"))
        if st.button("Delete file from index", key=k("rag_del_btn")):
            base_url_eff = st.session_state.cfg_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            n = delete_by_filename(base_url_eff, del_name)
            if n:
                st.success(f"Deletion requested for filename='{del_name}'.")
            else:
                st.warning("Nothing deleted (check exact filename).")

with st.expander("RAG: Ask", expanded=False):
    rag_q = st.text_input("Question (retrievalâ€‘augmented)", key=k("rag_q"))
    k_docs = st.slider("Top-K passages", 1, 10, 5, 1, key=k("rag_k"))
    # Optional filename filter
    try:
        base_url_eff = st.session_state.cfg_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        fn_options = ["(all)"] + list_filenames(base_url_eff)
    except Exception:
        fn_options = ["(all)"]
    sel_fn = st.selectbox("Filter by filename (optional)", options=fn_options, index=0, key=k("rag_filter_fn"))
    if st.button("Retrieve + Answer", key=k("rag_answer_btn")):
        base_url_eff = st.session_state.cfg_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            with st.spinner("Retrieving..."):
                hits = similarity_search_filtered(
                    base_url_eff,
                    rag_q,
                    k=k_docs,
                    filename=None if sel_fn == "(all)" else sel_fn,
                )
            if not hits:
                st.info("No relevant passages found. Try rephrasing or upload documents.")
            context_blocks = []
            for i, h in enumerate(hits, 1):
                meta = h.get("metadata", {})
                fn = meta.get("filename", "")
                context_blocks.append(f"[Doc {i} {fn}]\n{h['page_content']}")
            context = "\n\n".join(context_blocks)
            sys_msg = (
                "You are a grounded assistant. Use ONLY the provided context to answer. "
                "If the answer is not in the context, say you don't have enough information. "
                "Cite doc numbers like [Doc 2] when referencing."
            )
            messages = [
                SystemMessage(content=sys_msg),
                HumanMessage(content=f"Question: {rag_q}\n\nContext:\n{context}"),
            ]
            alias = "rag_worker" if "rag_worker" in MODELS else "chat"
            llm = st.session_state.llm_map.get(alias) or next(iter(st.session_state.llm_map.values()))
            with st.spinner("Answering..."):
                out = llm.invoke(
                    messages,
                    config=_build_run_config(
                        "rag_answer",
                        name="rag_answer",
                        meta={
                            "k": k_docs,
                            "hits": len(hits),
                            "files": [h.get("metadata", {}).get("filename", "") for h in hits],
                        },
                    ),
                )
            st.markdown(out.content)
            with st.expander("Retrieved passages", expanded=False):
                for i, h in enumerate(hits, 1):
                    meta = h.get("metadata", {})
                    fn = meta.get("filename", "")
                    st.markdown(f"**Doc {i}** â€” {fn}")
                    st.text(h["page_content"][:2000])
        except Exception as e:
            st.error(f"RAG failed: {e}")

# ==========================
@st.cache_data(show_spinner=False)
def _load_examples(_mtime: float) -> Dict[str, List[str]]:
    """Load examples from YAML. Cache invalidates when file mtime changes."""
    try:
        with open("prompts/examples.yaml", "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        clean: Dict[str, List[str]] = {}
        for k, v in data.items():
            if isinstance(v, list):
                clean[k] = [str(x) for x in v]
        return clean
    except Exception:
        return {}

with st.expander("Prompt Examples (by mode)", expanded=False):
    try:
        mtime = os.path.getmtime("prompts/examples.yaml")
    except Exception:
        mtime = 0.0
    examples = _load_examples(mtime)
    if not examples:
        st.info("No examples loaded. Add prompts to `prompts/examples.yaml`.")
    else:
        modes = sorted(examples.keys())
        sel = st.selectbox("Mode", options=modes, index=0, key=k("examples_mode"))
        for i, p in enumerate(examples.get(sel, []), 1):
            # Derive a compact title from the first non-empty line
            try:
                first_line = next((ln.strip() for ln in p.splitlines() if ln.strip()), f"Example {i}")
                title = first_line[:100]
            except Exception:
                title = f"Example {i}"

            with st.expander(f"{i}. {title}", expanded=False):
                st.code(p, language="text")
                # Toolbar
                cols = st.columns([1,1,1,6])
                # Use
                with cols[0]:
                    if st.button("Use", key=k(f"use_example_{sel}_{i}")):
                        st.session_state.pending_input = p
                        st.rerun()
                # Copy
                with cols[1]:
                    try:
                        _js = json.dumps(p)
                    except Exception:
                        _js = json.dumps(str(p))
                    components.html(
                        f"""
                        <button style='margin-top:6px;padding:6px 10px;border:1px solid #bbb;border-radius:6px;background:#fafafa;cursor:pointer' 
                                onclick='navigator.clipboard.writeText({_js}); this.innerText="Copied"; setTimeout(()=>this.innerText="Copy", 1500);'>
                          Copy
                        </button>
                        """,
                        height=40,
                    )
                # Customize toggle
                with cols[2]:
                    show_custom = st.toggle("Customize", key=k(f"customize_toggle_{sel}_{i}"), value=False)

                if show_custom:
                    st.markdown("Fill placeholders below; leave blank to remove them.")
                    # Extract placeholders from [] and <> patterns
                    sq = set(re.findall(r"\[([^\[\]\n]+)\]", p))
                    ang = set(re.findall(r"<([^<>\n]+)>", p))
                    tokens = [(t, "square") for t in sorted(sq)] + [(t, "angle") for t in sorted(ang)]
                    inputs: Dict[str, str] = {}
                    for t, kind in tokens:
                        ph_key = k(f"ph_{sel}_{i}_{kind}_{abs(hash(t))%10_000}")
                        multiline = any(w in t.lower() for w in ["text", "notes", "document", "code", "description"])
                        label = f"{t}"
                        if multiline:
                            inputs[t] = st.text_area(label, key=ph_key, height=120)
                        else:
                            inputs[t] = st.text_input(label, key=ph_key)
                    if st.button("Preview & Use", key=k(f"preview_use_{sel}_{i}")):
                        filled = p
                        for t in sq:
                            filled = filled.replace(f"[{t}]", inputs.get(t, ""))
                        for t in ang:
                            filled = filled.replace(f"<{t}>", inputs.get(t, ""))
                        st.session_state.pending_input = filled
                        st.rerun()

# ==========================
# 10) Render chat history
# ==========================
for msg in st.session_state.history:
    role = "user" if isinstance(msg, HumanMessage) else ("assistant" if isinstance(msg, AIMessage) else "system")
    with st.chat_message(role):
        st.write(msg.content)

# ==========================
# 11) Streaming with soft timeout AFTER first token + stop
# ==========================
def _build_run_config(route_alias: str, *, name: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    tags = ["router-app", route_alias]
    extra = [t.strip() for t in st.session_state.get("cfg_ls_tags", "").split(",") if t.strip()]
    tags.extend(extra)
    md: Dict[str, Any] = {"mode": st.session_state.mode}
    if meta:
        try:
            md.update({k: v for k, v in meta.items() if v is not None})
        except Exception:
            pass
    cfg: Dict[str, Any] = {"tags": tags, "metadata": md}
    if name:
        cfg["run_name"] = name
    return cfg

def stream_reply(eff_history: List[AnyMessage], chosen_mode: str, use_soft: bool, budget_seconds: int) -> Dict[str, Any]:
    """
    Streams tokens from the resolved model.
    Soft timeout starts AFTER the first token is received.
    Returns {"reply": str, "route": str, "stopped": bool, "elapsed": float}.
    """
    route = resolve_route(eff_history, chosen_mode)
    llm = st.session_state.llm_map[route]
    run_cfg = _build_run_config(route)

    output_area = st.empty()
    prog = st.progress(0)
    info = st.empty()

    assembled = ""
    token_count = 0
    st.session_state.stop_flag = False  # reset

    request_start = time.time()
    first_token_time: Optional[float] = None
    deadline_after_first: Optional[float] = None

    def update_progress():
        pct = 0.0 if token_count == 0 else min(0.95, token_count / 150.0)
        prog.progress(int(pct * 100))

    last_meta: Dict[str, Any] | None = None

    for chunk in llm.stream(eff_history, config=run_cfg):
        if st.session_state.stop_flag:
            info.write("Stopped by user.")
            return {"reply": assembled, "route": route, "stopped": True, "elapsed": time.time() - request_start}

        piece = getattr(chunk, "content", None)
        if piece:
            assembled += str(piece)
            token_count += 1
            output_area.write(assembled)
            update_progress()
        try:
            rm = getattr(chunk, "response_metadata", None)
            if isinstance(rm, dict) and rm:
                last_meta = rm
        except Exception:
            pass
            now = time.time()

            if first_token_time is None:
                first_token_time = now
                if use_soft:
                    deadline_after_first = first_token_time + budget_seconds

            if use_soft and deadline_after_first is not None and now > deadline_after_first:
                info.write("Soft timeout reached (after first token).")
                return {"reply": assembled, "route": route, "stopped": True, "elapsed": now - request_start}

            elapsed_total = int(now - request_start)
            if first_token_time is None:
                info.write(f"Waiting for first token... Elapsed: {elapsed_total}s")
            else:
                elapsed_after_first = int(now - first_token_time)
                info.write(f"Tokens: {token_count} - Elapsed total: {elapsed_total}s - Since first token: {elapsed_after_first}s")

    prog.progress(100)
    total_elapsed_s = int(time.time() - request_start)
    try:
        inp = (last_meta or {}).get("prompt_eval_count") or (last_meta or {}).get("input_tokens") or 0
        outp = (last_meta or {}).get("eval_count") or (last_meta or {}).get("output_tokens") or 0
        tot = (last_meta or {}).get("total_tokens")
        if not isinstance(tot, int):
            tot = (int(inp) if isinstance(inp, int) else 0) + (int(outp) if isinstance(outp, int) else 0)
        model_name = (last_meta or {}).get("model_name") or (last_meta or {}).get("model") or MODELS.get(route, "")
        info.write(f"Model: {model_name} | Time: {total_elapsed_s}s | Tokens — in: {inp}, out: {outp}, total: {tot}")
    except Exception:
        model_name = MODELS.get(route, "")
        info.write(f"Model: {model_name} | Time: {total_elapsed_s}s | Tokens emitted: {token_count}")
    return {"reply": assembled, "route": route, "stopped": False, "elapsed": time.time() - request_start}

# ==========================
# 12) Non-streaming invoke
# ==========================
def invoke_reply(eff_history: List[AnyMessage], chosen_mode: str) -> Dict[str, Any]:
    route = resolve_route(eff_history, chosen_mode)
    llm = st.session_state.llm_map[route]
    start_time = time.time()
    with st.spinner("Thinking..."):
        resp = llm.invoke(eff_history, config=_build_run_config(route, name="router_invoke"))
    return {"reply": resp.content, "route": route, "stopped": False, "elapsed": time.time() - start_time}

# ==========================
# 13) Input and response
# ==========================
user_input = st.chat_input("Write a message... (type /clear to reset)", key=k("chat_input"))
if not user_input and st.session_state.get("pending_input"):
    user_input = st.session_state.pending_input
    st.session_state.pending_input = None
if user_input:
    if user_input.strip() == "/clear":
        st.session_state.history = []
        st.session_state.route_used = None
        st.rerun()

    # Decide route before mutating history
    prior_history = st.session_state.history.copy()
    eff_history = ensure_system_prompt(prior_history, sys_prompt)
    resolved_route = resolve_route(eff_history + [HumanMessage(content=user_input)], st.session_state.mode)

    # Render user message
    st.session_state.history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.write(user_input)

    # Assistant turn
    with st.chat_message("assistant"):
        if st.session_state.use_tools:
            # Use explicit tools model from sidebar
            alias = st.session_state.get("cfg_tools_alias", "router_light")
            # Build a minimal relevant toolset per query to improve reliability
            tools_subset = select_tools_for_query(user_input)
            llm_for_tools = st.session_state.llm_map.get(alias) or next(iter(st.session_state.llm_map.values()))
            # Strict only for purely time/date toolsets, if enabled
            tool_names = {getattr(t, "name", getattr(t, "__name__", "")) for t in tools_subset}
            is_time_only = tool_names.issubset({"now", "today", "time_in"}) and len(tool_names) > 0
            strict_flag = bool(st.session_state.get("cfg_tools_strict", False) and is_time_only)
            executor = build_tools_agent(
                llm_for_tools,
                max_iterations=st.session_state.get("cfg_tools_iterations", 10),
                verbose=st.session_state.get("cfg_tools_verbose", False),
                tools_override=tools_subset,
                strict=strict_flag,
            )
            data = invoke_tools_agent(executor, eff_history, user_input, run_config=_build_run_config(f"{alias}+tools"))
            # Show agent steps
            steps = data.get("steps", [])
            if steps:
                with st.expander("Agent steps", expanded=True):
                    for i, s in enumerate(steps, 1):
                        if "raw" in s:
                            st.write(f"Step {i}: {s['raw']}")
                        else:
                            st.markdown(f"- Step {i} â€¢ Tool: `{s['tool']}`")
                            st.caption(f"Input: {s.get('input','')}")
                            obs = s.get("observation", "")
                            if len(obs) > 1200:
                                obs = obs[:1200] + "\n...[truncated]"
                            st.text(obs)
                # If we hit configured step limit, surface a hint
                if len(steps) >= st.session_state.get("cfg_tools_iterations", 10):
                    st.info("Reached max tool iterations. Refine the query or increase 'Max tool iterations' in sidebar.")
            # Save steps to logs if enabled
            if st.session_state.get("cfg_tools_save_logs", False):
                try:
                    os.makedirs("logs", exist_ok=True)
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    fname = f"logs/agent_steps_{stamp}.json"
                    payload = {
                        "timestamp": stamp,
                        "input": user_input,
                        "model_alias": alias,
                        "settings": {
                            "iterations": st.session_state.get("cfg_tools_iterations"),
                            "verbose": st.session_state.get("cfg_tools_verbose"),
                            "web_get_timeout": st.session_state.get("cfg_web_get_timeout"),
                            "web_search_timeout": st.session_state.get("cfg_web_search_timeout"),
                        },
                        "steps": steps,
                        "output": data.get("output", ""),
                    }
                    with open(fname, "w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
                    st.caption(f"Saved agent steps to {fname}")
                except Exception as e:
                    st.caption(f"Log save failed: {e}")
            reply = data.get("output", "")
            st.write(reply)
            result = {"route": f"{alias}+tools"}
        else:
            if st.session_state.use_streaming:
                result = stream_reply(
                    ensure_system_prompt(st.session_state.history, sys_prompt),
                    st.session_state.mode,
                    use_soft=use_soft_timeout,
                    budget_seconds=soft_timeout_seconds,
                )
                reply = result["reply"]
            else:
                result = invoke_reply(ensure_system_prompt(st.session_state.history, sys_prompt), st.session_state.mode)
                reply = result["reply"]
                st.write(reply)

    st.session_state.route_used = result.get("route", "-")
    st.session_state.history.append(AIMessage(content=reply))

# ==========================
# 14) Debug
# ==========================
with st.expander("Debug / state", expanded=False):
    st.write("Current route:", st.session_state.route_used or "-")
    st.write("History length:", len(st.session_state.history))
    st.json({
        "mode": st.session_state.mode,
        "base_url": base_url,
        "temperature": temperature,
        "models": MODELS,
        "request_timeout": request_timeout,
        "ocr_model": st.session_state.get("cfg_ocr_model") or os.getenv("OLLAMA_OCR_MODEL", "llava:34b"),
        "ocr_options": {
            "force_pdf_ocr": st.session_state.get(k("ocr_any_force")),
            "pdf_render_dpi": st.session_state.get(k("ocr_any_dpi")),
            "max_pdf_pages": st.session_state.get(k("ocr_any_max_pages")),
        },
        "ocr_prompt_preview": (st.session_state.get("cfg_ocr_prompt") or "")[:160],
        "use_tools": st.session_state.use_tools,
        "tools_alias": st.session_state.get("cfg_tools_alias"),
        "tools_iterations": st.session_state.get("cfg_tools_iterations"),
        "tools_verbose": st.session_state.get("cfg_tools_verbose"),
    })


