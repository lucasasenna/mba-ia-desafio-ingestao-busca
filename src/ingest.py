import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Variáveis exigidas por provedor: chave de API, modelo de embeddings e LLM.
PROVIDERS = {
    "gemini": ("GOOGLE_API_KEY", "GOOGLE_EMBEDDING_MODEL", "GOOGLE_LLM_MODEL"),
    "openai": ("OPENAI_API_KEY", "OPENAI_EMBEDDING_MODEL", "OPENAI_LLM_MODEL"),
}


class ConfigError(Exception):
    """Erro de configuração ou de ambiente com mensagem pronta para o usuário."""


def require_env(name):
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(f"A variável {name} não está definida no .env.")
    return value


def provider_settings():
    """Retorna (api_key, embedding_model, llm_model) do provedor escolhido."""
    if LLM_PROVIDER not in PROVIDERS:
        raise ConfigError(
            f"LLM_PROVIDER inválido: '{LLM_PROVIDER}'. Use 'gemini' ou 'openai'."
        )
    key_var, embedding_var, llm_var = PROVIDERS[LLM_PROVIDER]
    api_key = os.getenv(key_var, "").strip()
    if not api_key:
        raise ConfigError(
            f"Chave de API ausente: defina {key_var} no .env "
            f"(provedor atual: {LLM_PROVIDER})."
        )
    return api_key, require_env(embedding_var), require_env(llm_var)


def get_embeddings():
    api_key, model, _ = provider_settings()
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        # transport="rest" evita o traceback do cliente gRPC assíncrono ao sair.
        return GoogleGenerativeAIEmbeddings(
            model=model, google_api_key=api_key, transport="rest"
        )

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=model, api_key=api_key)


def check_database():
    """Falha cedo, com mensagem clara, se o Postgres não estiver acessível."""
    url = require_env("DATABASE_URL")
    try:
        engine = create_engine(url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
    except OperationalError as exc:
        raise ConfigError(
            "Não foi possível conectar ao banco de dados em DATABASE_URL. "
            "Verifique se o Postgres está no ar (docker compose up -d).\n"
            f"Detalhe: {str(exc.orig).splitlines()[0]}"
        ) from exc
    return url


def get_vector_store(embeddings):
    from langchain_postgres import PGVector

    return PGVector(
        embeddings=embeddings,
        collection_name=require_env("PG_VECTOR_COLLECTION_NAME"),
        connection=check_database(),
        use_jsonb=True,
    )


def resolve_pdf_path():
    path = Path(PDF_PATH)
    if not path.is_absolute() and not path.exists():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise ConfigError(f"PDF não encontrado: {PDF_PATH} (verifique PDF_PATH no .env).")
    return path


def ingest_pdf():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    pdf_path = resolve_pdf_path()
    embeddings = get_embeddings()
    store = get_vector_store(embeddings)

    print(f"Carregando {pdf_path.name}...")
    pages = PyPDFLoader(str(pdf_path)).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = [c for c in splitter.split_documents(pages) if c.page_content.strip()]
    if not chunks:
        raise ConfigError("Nenhum texto foi extraído do PDF.")

    # Reingestão idempotente: apaga a collection e recria antes de inserir.
    store.delete_collection()
    store.create_collection()

    ids = [f"chunk-{i}" for i in range(len(chunks))]
    print(f"Gerando embeddings e gravando {len(chunks)} chunks ({len(pages)} páginas)...")
    store.add_documents(chunks, ids=ids)

    print(
        f"Ingestão concluída: {len(chunks)} chunks gravados na collection "
        f"'{COLLECTION_NAME}'."
    )


def main():
    try:
        ingest_pdf()
    except ConfigError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nIngestão interrompida.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"Erro durante a ingestão: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
