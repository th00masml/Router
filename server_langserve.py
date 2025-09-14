from fastapi import FastAPI
from langserve import add_routes

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI  # optional

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server exposing runnables via LangServe",
)

# --- Models ---
ollama_model = ChatOllama(model="mistral-small:24b", temperature=0)  # your local model
# openai_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)       # optional

# --- Prompts / chains ---
prompt_essay = ChatPromptTemplate.from_template("Write a 100-word essay about {topic}.")
prompt_song  = ChatPromptTemplate.from_template("Write a 100-word song about {topic}.")

essay_chain = prompt_essay | ollama_model | StrOutputParser()
song_chain  = prompt_song  | ollama_model | StrOutputParser()  # or a different model

# --- Expose routes ---
add_routes(app, ollama_model, path="/ollama")   # plain chat: input is string
# add_routes(app, openai_model, path="/openai") # optional

add_routes(app, essay_chain, path="/essay")     # input: {"topic": "..."}
add_routes(app, song_chain,  path="/song")      # input: {"topic": "..."}

from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    # Walk all paths and remove any header parameter named "x-session-id"
    for path, methods in schema.get("paths", {}).items():
        for method, op in methods.items():
            if not isinstance(op, dict):
                continue
            params = op.get("parameters", [])
            filtered = []
            for p in params:
                # Sometimes name can be 'x-session-id' or 'x-session-id'.lower()
                if not (p.get("in") == "header" and p.get("name", "").lower() == "x-session-id"):
                    filtered.append(p)
            op["parameters"] = filtered
    app.openapi_schema = schema
    return app.openapi_schema

_original_openapi = app.openapi

