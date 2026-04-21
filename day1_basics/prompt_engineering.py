"""
Day 1 - Script 2: Prompt Engineering Techniques
Demonstrates zero-shot, few-shot, and chain-of-thought prompting using Groq LLM.
Run: python day1_basics/prompt_engineering.py
"""

import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("[ERROR] GROQ_API_KEY not found. Copy .env.example to .env and add your key.")
    sys.exit(1)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    api_key=GROQ_API_KEY,
)

SAMPLE_TEXT = """
Annual Revenue grew from $150,000 in January to $310,000 in December 2024.
Total expenses were $1,387,000 for the year. Net profit was $1,328,000.
Product A leads in the North region. Product C has the highest growth rate at 45%.
"""


def zero_shot_prompt():
    """Technique 1: Zero-shot - No examples, just a direct instruction."""
    print("\n" + "=" * 60)
    print("TECHNIQUE 1: Zero-Shot Prompting")
    print("=" * 60)

    messages = [
        SystemMessage(content="You are a financial analyst. Be concise."),
        HumanMessage(content=f"Summarize the key financial insights from this data:\n\n{SAMPLE_TEXT}"),
    ]
    response = llm.invoke(messages)
    print(response.content)


def few_shot_prompt():
    """Technique 2: Few-shot - Provide examples to guide the output format."""
    print("\n" + "=" * 60)
    print("TECHNIQUE 2: Few-Shot Prompting")
    print("=" * 60)

    messages = [
        SystemMessage(content="You are a financial analyst who extracts structured insights."),
        HumanMessage(
            content="""Here are examples of how to extract insights:

Example 1:
Data: Q1 revenue was $500K, Q2 was $600K.
Insight: Revenue grew 20% from Q1 to Q2, showing positive momentum.

Example 2:
Data: Expenses rose from $200K to $300K while revenue stayed flat at $400K.
Insight: Profit margin compressed by 25 percentage points - a red flag requiring cost review.

Now extract insights from:
Data: """
            + SAMPLE_TEXT
        ),
    ]
    response = llm.invoke(messages)
    print(response.content)


def chain_of_thought_prompt():
    """Technique 3: Chain-of-Thought - Ask LLM to reason step by step."""
    print("\n" + "=" * 60)
    print("TECHNIQUE 3: Chain-of-Thought Prompting")
    print("=" * 60)

    messages = [
        SystemMessage(content="You are a financial analyst. Think step by step before answering."),
        HumanMessage(
            content=f"""Analyze this financial data and answer: Is the company financially healthy?

Data:
{SAMPLE_TEXT}

Let's think step by step:
1. First, evaluate revenue trend
2. Then, evaluate expense vs profit ratio
3. Then, identify any risks
4. Finally, give an overall health assessment."""
        ),
    ]
    response = llm.invoke(messages)
    print(response.content)


def role_based_prompt():
    """Technique 4: Role-based prompting - Assign a specific expert persona."""
    print("\n" + "=" * 60)
    print("TECHNIQUE 4: Role-Based Prompting")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a CFO with 20 years of experience preparing board presentations. "
                "You communicate complex financial data in simple, clear bullet points "
                "suitable for non-financial stakeholders."
            )
        ),
        HumanMessage(
            content=f"Prepare a 3-bullet executive summary of this data for the board:\n\n{SAMPLE_TEXT}"
        ),
    ]
    response = llm.invoke(messages)
    print(response.content)


if __name__ == "__main__":
    print("Running all 4 Prompt Engineering techniques...")
    zero_shot_prompt()
    few_shot_prompt()
    chain_of_thought_prompt()
    role_based_prompt()
    print("\n[SUCCESS] Prompt engineering demo complete!")
