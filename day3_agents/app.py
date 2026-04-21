"""
Day 3 - Multi-Agent Chatbot UI
Streamlit app with 3 agent modes: Q&A, Summarize, MCQ Generator.
Run: streamlit run day3_agents/app.py
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from crewai import Crew, Process
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from day3_agents.tasks import create_qa_task, create_summary_task, create_mcq_task

CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

st.set_page_config(
    page_title="Financial AI Assistant",
    page_icon="",
    layout="wide",
)

st.markdown("""
    <style>
    .main-header { font-size: 2rem; font-weight: bold; color: #1f77b4; }
    .agent-badge { padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: bold; }
    .qa-badge { background-color: #d4edda; color: #155724; }
    .summary-badge { background-color: #cce5ff; color: #004085; }
    .mcq-badge { background-color: #fff3cd; color: #856404; }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_retriever():
    if not os.path.exists(CHROMA_PATH):
        return None
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 6})


def get_context(retriever, query: str, source_filter: str = None) -> str:
    docs = retriever.invoke(query)
    if source_filter:
        docs = [d for d in docs if source_filter.lower() in d.metadata.get("source", "").lower()]
    if not docs:
        docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)


def run_agent(task):
    try:
        crew = Crew(
            agents=[task.agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )
        result = crew.kickoff()
        return str(result)
    except Exception as e:
        return f"[Agent Error] {e}"


st.markdown('<p class="main-header"> Financial AI Assistant</p>', unsafe_allow_html=True)
st.markdown("Powered by **Groq LLaMA 3.3** + **CrewAI Agents** + **RAG Pipeline**")
st.divider()

retriever = load_retriever()

if retriever is None:
    st.error("ChromaDB not found. Run `python day2_rag/ingest.py` first.")
    st.stop()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Select Agent Mode")
    mode = st.radio(
        "Choose what you want to do:",
        options=[" Question Answering", " Summarize Document", " Generate MCQs"],
        index=0,
    )
    st.divider()

    if mode == " Question Answering":
        st.markdown('<span class="agent-badge qa-badge">QA Agent Active</span>', unsafe_allow_html=True)
        st.info("Ask any question about the financial documents.")
    elif mode == " Summarize Document":
        st.markdown('<span class="agent-badge summary-badge">Summarizer Agent Active</span>', unsafe_allow_html=True)
        st.info("Get a structured executive summary of the documents.")
    else:
        st.markdown('<span class="agent-badge mcq-badge">MCQ Agent Active</span>', unsafe_allow_html=True)
        st.info("Generate multiple choice questions for learning.")

with col2:
    st.subheader("Output")

    if mode == " Question Answering":
        question = st.text_input("Your question:", placeholder="e.g. What was HDFC Bank's net profit in Q1 FY26?")
        if st.button("Get Answer", type="primary") and question:
            with st.spinner("QA Agent is thinking..."):
                context = get_context(retriever, question, source_filter="q1fy26")
                task = create_qa_task(question, context)
                answer = run_agent(task)
            st.success("Answer:")
            st.markdown(answer)

    elif mode == " Summarize Document":
        if st.button("Generate Summary", type="primary"):
            with st.spinner("Summarizer Agent is working..."):
                context = get_context(retriever, "HDFC Bank revenue profit NII CASA asset quality capital adequacy", source_filter="q1fy26")
                task = create_summary_task(context)
                summary = run_agent(task)
            st.success("Executive Summary:")
            st.markdown(summary)

    else:
        num_q = st.slider("Number of MCQs to generate:", min_value=3, max_value=10, value=5)
        if st.button("Generate MCQs", type="primary"):
            with st.spinner(f"MCQ Agent is generating {num_q} questions..."):
                context = get_context(retriever, "HDFC Bank deposits advances NPA profit capital ratio subsidiaries", source_filter="q1fy26")
                task = create_mcq_task(context, num_questions=num_q)
                mcqs = run_agent(task)
            st.success("Generated MCQs:")
            st.markdown(mcqs)

st.divider()
st.caption("AI Training PoC - Day 3 - Multi-Agent System with CrewAI")
