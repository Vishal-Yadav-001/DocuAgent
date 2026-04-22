"""
Day 2 - Script 2: RAG Query Chain
Ask questions about your documents using retrieval-augmented generation.
Run: python day2_rag/query.py
"""

import os
import sys
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("[ERROR] GROQ_API_KEY not found. Add it to your .env file.")
    sys.exit(1)


def load_retriever():
    if not os.path.exists(CHROMA_PATH):
        print("[ERROR] ChromaDB not found. Run: python day2_rag/ingest.py first!")
        sys.exit(1)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )
    return vectorstore.as_retriever(search_kwargs={"k": 6})


def build_rag_chain(retriever):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=GROQ_API_KEY,
    )

    prompt = ChatPromptTemplate.from_template("""
You are a financial analyst assistant. Answer the question using ONLY the context provided.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs):
        parts = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[Source: {source}]\n{doc.page_content}")
        return "\n\n".join(parts)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def run_demo_queries(chain, retriever):
    questions = [
        "What is HDFC Bank's net revenue or net interest income for Q1 FY26?",
        "What is HDFC Bank's net profit or PAT for Q1 FY26?",
        "What is HDFC Bank's GNPA or asset quality in Q1 FY26?",
        "What are HDFC Bank's key business highlights for Q1 FY26?",
        "What is HDFC Bank's capital adequacy ratio or CRAR?",
    ]

    for q in questions:
        print(f"\n{'=' * 60}")
        print(f"Q: {q}")
        print(f"{'=' * 60}")
        docs = retriever.invoke(q)
        sources = list({d.metadata.get('source', '?') for d in docs})
        answer = chain.invoke(q)
        print(f"A: {answer}")
        print(f"  [Sources: {', '.join(sources)}]")


if __name__ == "__main__":
    print("Loading RAG pipeline...")
    retriever = load_retriever()
    chain, retriever = build_rag_chain(retriever)
    print("[SUCCESS] RAG chain ready!\n")

    run_demo_queries(chain, retriever)

    print("\n" + "=" * 60)
    print("Interactive mode - type your question (or 'quit' to exit)")
    print("=" * 60)
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        try:
            docs = retriever.invoke(question)
            sources = list({d.metadata.get('source', '?') for d in docs})
            answer = chain.invoke(question)
            print(f"Answer: {answer}")
            print(f"[Sources: {', '.join(sources)}]")
        except Exception as e:
            print(f"[ERROR] Could not get answer: {e}")
