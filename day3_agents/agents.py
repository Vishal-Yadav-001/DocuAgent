"""
Day 3 - Agents Definition
Defines three CrewAI agents: QA Agent, Summarizer Agent, MCQ Generator Agent.
"""

import os
import sys
from dotenv import load_dotenv
from crewai import Agent

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    print("[ERROR] GROQ_API_KEY not found. Add it to your .env file.")
    sys.exit(1)

# CrewAI uses LiteLLM internally - the "groq/" prefix tells LiteLLM
# which provider to route to. GROQ_API_KEY is read from env automatically.
GROQ_MODEL = "groq/llama-3.3-70b-versatile"

qa_agent = Agent(
    role="Financial Question Answering Expert",
    goal="Answer specific questions about financial documents accurately and concisely.",
    backstory=(
        "You are a seasoned financial analyst with 15 years of experience reading "
        "annual reports, balance sheets, and financial statements. You provide precise, "
        "data-backed answers to financial questions."
    ),
    llm=GROQ_MODEL,
    verbose=True,
    allow_delegation=False,
)

summarizer_agent = Agent(
    role="Executive Financial Summarizer",
    goal="Create clear, structured summaries of financial documents for stakeholders.",
    backstory=(
        "You are an expert at distilling complex financial documents into concise, "
        "readable summaries. You communicate key takeaways in plain language that "
        "both technical and non-technical audiences can understand."
    ),
    llm=GROQ_MODEL,
    verbose=True,
    allow_delegation=False,
)

mcq_agent = Agent(
    role="Financial Assessment Creator",
    goal="Generate meaningful multiple choice questions (MCQs) with correct answers from financial content.",
    backstory=(
        "You are an experienced trainer who creates educational assessments. "
        "You craft clear, unambiguous MCQs that test comprehension of financial concepts "
        "and data. Each question has 4 options with exactly one correct answer."
    ),
    llm=GROQ_MODEL,
    verbose=True,
    allow_delegation=False,
)
