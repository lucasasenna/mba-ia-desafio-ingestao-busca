# Ingestão e Busca Semântica com LangChain e Postgres

Solução do Desafio 1 do MBA em Engenharia de Software com IA (Full Cycle). Enunciado original: https://github.com/devfullcycle/mba-ia-desafio-ingestao-busca

Um RAG mínimo de linha de comando:

- **Ingestão** (`src/ingest.py`): lê o `document.pdf`, divide o texto em chunks de 1000 caracteres com overlap de 150, gera os embeddings e grava tudo no PostgreSQL com pgVector.
- **Busca** (`src/search.py`): vetoriza a pergunta, busca os 10 chunks mais parecidos (`similarity_search_with_score(query, k=10)`), monta o prompt do enunciado e chama a LLM com temperatura 0.
- **Chat** (`src/chat.py`): CLI interativa que responde usando somente o conteúdo do PDF.

## Tecnologias

- Python, LangChain, PostgreSQL + pgVector, Docker Compose
- Provedor padrão: **Google Gemini**. OpenAI também é suportado: troque `LLM_PROVIDER` no `.env`.

### Modelos escolhidos

Os nomes dos modelos vêm do `.env`, nenhum está fixo no código. Os padrões são os mais leves e baratos indicados na documentação oficial em novembro de 2026:

| Provedor | Embeddings | LLM |
|---|---|---|
| Gemini (padrão) | `models/gemini-embedding-001` (3072 dimensões) | `gemini-3.1-flash-lite` |
| OpenAI | `text-embedding-3-small` (1536 dimensões) | `gpt-6-luna` |

## Pré-requisitos

- Python 3.11 a 3.13 (testado com 3.12). As versões fixadas no `requirements.txt` ainda não têm pacotes prontos para Python 3.14 ou mais novo.
- Docker com Docker Compose
- Chave de API do provedor escolhido (Gemini: https://aistudio.google.com/apikey)

## Como executar

1. Crie e ative o ambiente virtual:

```bash
python3 -m venv venv && source venv/bin/activate
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Crie o `.env` e preencha a chave de API (`GOOGLE_API_KEY` ou `OPENAI_API_KEY`):

```bash
cp .env.example .env
```

4. Suba o banco de dados:

```bash
docker compose up -d
```

5. Rode a ingestão do PDF:

```bash
python src/ingest.py
```

O script mostra quantos chunks foram gravados. Pode rodar quantas vezes quiser: a collection é apagada e recriada a cada ingestão, então os chunks não se duplicam.

6. Rode o chat:

```bash
python src/chat.py
```

Para sair, digite `sair` ou `exit`, ou use Ctrl+C.

## Exemplo

```
Faça sua pergunta:
> Qual o faturamento da empresa Alfa Energia S.A.?

PERGUNTA: Qual o faturamento da empresa Alfa Energia S.A.?
RESPOSTA: O faturamento da empresa Alfa Energia S.A. é R$ 722.875.391,46.

Faça sua pergunta:
> Qual é a capital da França?

PERGUNTA: Qual é a capital da França?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

## Variáveis de ambiente

| Variável | Descrição |
|---|---|
| `LLM_PROVIDER` | `gemini` ou `openai` |
| `GOOGLE_API_KEY` / `OPENAI_API_KEY` | Chave do provedor escolhido |
| `GOOGLE_EMBEDDING_MODEL` / `OPENAI_EMBEDDING_MODEL` | Modelo de embeddings |
| `GOOGLE_LLM_MODEL` / `OPENAI_LLM_MODEL` | Modelo de LLM |
| `DATABASE_URL` | URL do Postgres, por exemplo `postgresql+psycopg://postgres:postgres@localhost:5432/rag` |
| `PG_VECTOR_COLLECTION_NAME` | Nome da collection no pgVector |
| `PDF_PATH` | Caminho do PDF (relativo à raiz do projeto ou absoluto) |

## ⚠️ Trocar o modelo de embeddings

Cada modelo de embeddings gera vetores de um tamanho diferente (por exemplo, 3072 no Gemini e 1536 na OpenAI). Vetores de dimensões diferentes não podem ser comparados. Se você trocar o modelo de embeddings ou o provedor, apague os dados antigos e refaça a ingestão do zero. Há duas formas:

- Rodar `python src/ingest.py` de novo, que apaga e recria a collection configurada em `PG_VECTOR_COLLECTION_NAME`, ou trocar `PG_VECTOR_COLLECTION_NAME` para um nome novo antes de rodar.
- Apagar o volume do banco e começar do zero:

```bash
docker compose down -v && docker compose up -d && python src/ingest.py
```

## Porta 5432 ocupada

Se outro Postgres já usa a porta 5432, o `docker compose up -d` falha com `port is already allocated`. Pare o outro serviço ou publique o banco em outra porta sem alterar o `docker-compose.yml`, criando um `docker-compose.override.yml`:

```yaml
services:
  postgres:
    ports: !override
      - "5433:5432"
```

Depois ajuste a porta no `.env`: `DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5433/rag`.

## Erros tratados

Os scripts param com uma mensagem clara, sem traceback, quando:

- o banco está fora do ar (lembram de rodar `docker compose up -d`);
- falta a chave de API ou alguma variável no `.env`;
- o PDF de `PDF_PATH` não existe;
- a collection está vazia ao abrir o chat (pedem para rodar a ingestão antes).
