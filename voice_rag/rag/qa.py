from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from voice_rag.config.settings import settings
from voice_rag.rag.ingest import load_existing_vector_store


def get_llm() -> ChatOllama:
    return ChatOllama(
        model=settings.ollama_model,
        temperature=0.1,
    )


RAG_PROMPT_TMPL = """आपण एक मदत करणारा सहायक आहात जो नेहमी मराठीत उत्तरे देतो.
फक्त दिलेल्या संदर्भाचा वापर करून उत्तर द्या. संदर्भामध्ये माहिती नसेल तर
ते स्पष्टपणे सांगा.

[संदर्भ]
{context}

[प्रश्न]
{question}

मराठीमध्ये संक्षिप्त आणि स्पष्ट उत्तर द्या:
"""

RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TMPL,
    input_variables=["context", "question"],
)


def get_qa_chain() -> RetrievalQA:
    vectorstore = load_existing_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = get_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": RAG_PROMPT},
        return_source_documents=False,
    )
    return qa_chain


def answer_question_marathi(question: str) -> str:
    qa = get_qa_chain()
    result = qa.invoke({"query": question})
    if isinstance(result, dict):
        return result.get("result") or result.get("output_text") or str(result)
    return str(result)
