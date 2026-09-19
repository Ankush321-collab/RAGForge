from functools import lru_cache
import os

from dotenv import load_dotenv
from httpx import HTTPStatusError
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_mistralai import ChatMistralAI

from ..retrieval.vector import (
    create_vector_store_from_pdf,
    similarity_search,
)

load_dotenv()

PDF_PATH = "data/raw/oops_notes.pdf"
LLM_MODEL_NAME = os.getenv("MISTRAL_MODEL", "ministral-8b-2512")
LLM_MAX_RETRIES = int(os.getenv("MISTRAL_MAX_RETRIES", "2"))

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer using only the provided PDF context. "
            "If the answer is not in the context, say you cannot find it in the PDF.",
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}\n\n"
            "Answer clearly and concisely.",
        ),
    ]
)


def format_documents(documents):
    """Format retrieved documents while retaining citation metadata."""
    return "\n\n".join(
        "[Document: {document_id}, Page: {page_number}, Chunk: {chunk_id}]\n{text}".format(
            document_id=document.metadata.get("document_id"),
            page_number=document.metadata.get("page_number"),
            chunk_id=document.metadata.get("chunk_id"),
            text=document.page_content,
        )
        for document in documents
    )


@lru_cache(maxsize=1)
def get_vector_store():
    """Create the persistent Chroma store only when the RAG chain is used."""
    return create_vector_store_from_pdf(
        PDF_PATH,
        collection_name="ragforge_chunks",
        persist_directory="data/chroma",
        chunk_size=500,
        overlap=50,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )


@lru_cache(maxsize=1)
def get_llm():
    """Create the Mistral chat model only when an answer is requested."""
    return ChatMistralAI(
        model_name=LLM_MODEL_NAME,
        temperature=0,
        max_retries=LLM_MAX_RETRIES,
    )


def build_rag_chain(vector_store, llm, k=5):
    """Build a LangChain Runnable retrieval-augmented generation chain."""
    retrieve = RunnableLambda(
        lambda question: similarity_search(vector_store, question, k=k)
    )
    return (
        {
            "context": retrieve | RunnableLambda(format_documents),
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )


def build_answer_chain(llm):
    """Build the Runnable that turns retrieved context into an answer."""
    return PROMPT | llm | StrOutputParser()


def answer_question(question: str, k=5) -> dict:
    """Retrieve context and generate a grounded answer."""
    documents = similarity_search(get_vector_store(), question, k=k)
    context = format_documents(documents)
    try:
        answer = build_answer_chain(get_llm()).invoke(
            {"context": context, "question": question}
        )
    except HTTPStatusError as error:
        if error.response.status_code == 429:
            raise RuntimeError(
                "Mistral rate limit reached. Wait for the quota window to reset "
                "or use another Mistral API key/model, then retry."
            ) from error
        raise

    return {
        "question": question,
        "context": context,
        "answer": answer,
        "citations": [document.metadata for document in documents],
    }


if __name__ == "__main__":
    result = answer_question("What is polymorphism?")
    print(result["answer"])
    print(result["citations"])
