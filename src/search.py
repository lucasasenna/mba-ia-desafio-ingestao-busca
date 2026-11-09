import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from sqlalchemy import create_engine, text

from ingest import (
    LLM_PROVIDER,
    ConfigError,
    check_database,
    get_embeddings,
    get_vector_store,
    provider_settings,
    require_env,
)

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

TOP_K = 10


def get_llm():
    api_key, _, model = provider_settings()
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # transport="rest" evita o traceback do cliente gRPC assíncrono ao sair.
        return ChatGoogleGenerativeAI(
            model=model, temperature=0, google_api_key=api_key, transport="rest"
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, temperature=0, api_key=api_key)


def ensure_collection_has_data():
    """Avisa para rodar a ingestão se a collection estiver vazia ou não existir."""
    collection = require_env("PG_VECTOR_COLLECTION_NAME")
    url = check_database()
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            count = conn.execute(
                text(
                    "SELECT count(e.id) FROM langchain_pg_collection c "
                    "JOIN langchain_pg_embedding e ON e.collection_id = c.uuid "
                    "WHERE c.name = :name"
                ),
                {"name": collection},
            ).scalar()
    except Exception:
        count = 0
    finally:
        engine.dispose()
    if not count:
        raise ConfigError(
            f"A collection '{collection}' está vazia ou não existe. "
            "Rode antes: python src/ingest.py"
        )


def search_prompt(question=None):
    """Monta a chain de RAG (busca k=10 + prompt + LLM).

    Sem argumento, devolve a chain para ser chamada com chain.invoke(pergunta).
    Com uma pergunta, devolve direto a resposta. Em caso de erro de
    configuração, imprime a causa e devolve None.
    """
    try:
        ensure_collection_has_data()
        store = get_vector_store(get_embeddings())
        llm = get_llm()
    except ConfigError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return None

    def build_context(pergunta):
        results = store.similarity_search_with_score(pergunta, k=TOP_K)
        return "\n\n".join(doc.page_content for doc, _score in results)

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = (
        {"contexto": RunnableLambda(build_context), "pergunta": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(str.strip)
    )

    if question is not None:
        return chain.invoke(question)
    return chain
