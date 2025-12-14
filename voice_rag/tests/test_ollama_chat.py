from langchain_community.chat_models import ChatOllama


def main():
    """Smoke test: call Ollama chat locally and print the response."""
    llm = ChatOllama(model="llama3", temperature=0.1)
    resp = llm.invoke("Say 'Hello from ARQ Voice Agent' in English.")
    print("LLM response:", resp.content)


if __name__ == "__main__":
    main()
