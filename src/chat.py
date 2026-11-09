import sys

from search import search_prompt

EXIT_COMMANDS = {"sair", "exit"}


def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print('Chat iniciado. Digite "sair" ou "exit" para encerrar.\n')
    try:
        while True:
            print("Faça sua pergunta:")
            question = input("> ").strip()
            if not question:
                continue
            if question.lower() in EXIT_COMMANDS:
                break

            print(f"\nPERGUNTA: {question}")
            try:
                answer = chain.invoke(question)
            except Exception as exc:
                print(f"Erro ao consultar a LLM: {exc}\n", file=sys.stderr)
                continue
            print(f"RESPOSTA: {answer}\n")
    except (KeyboardInterrupt, EOFError):
        print()

    print("Até mais!")


if __name__ == "__main__":
    main()
