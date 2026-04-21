"""
Day 3 - Tasks Definition
Creates tasks for each agent based on user input and document content.
"""

from crewai import Task
from day3_agents.agents import qa_agent, summarizer_agent, mcq_agent


def create_qa_task(question: str, context: str) -> Task:
    return Task(
        description=f"""
        Using the following financial document context, answer this question:

        QUESTION: {question}

        CONTEXT:
        {context}

        Provide a clear, accurate answer with specific numbers or facts from the context.
        If the information is not available in the context, say so clearly.
        """,
        expected_output="A direct, factual answer to the question with supporting data from the document.",
        agent=qa_agent,
    )


def create_summary_task(context: str) -> Task:
    return Task(
        description=f"""
        Create a comprehensive executive summary of the following financial document content.

        DOCUMENT CONTENT:
        {context}

        Your summary must include:
        1. Key financial highlights (revenue, expenses, profit)
        2. Top performing products/regions
        3. Major risks identified
        4. Strategic outlook
        5. Overall assessment (1-2 sentences)

        Keep it under 300 words. Use bullet points where appropriate.
        """,
        expected_output="A structured executive summary with 5 sections covering the financial document's key points.",
        agent=summarizer_agent,
    )


def create_mcq_task(context: str, num_questions: int = 5) -> Task:
    return Task(
        description=f"""
        Generate {num_questions} multiple choice questions (MCQs) based on this financial document content.

        DOCUMENT CONTENT:
        {context}

        Format each MCQ as:
        Q[number]. [Question text]
        A) [Option A]
        B) [Option B]
        C) [Option C]
        D) [Option D]
        Correct Answer: [Letter]
        Explanation: [Brief explanation]

        Make questions test understanding of key financial metrics, trends, and insights.
        """,
        expected_output=f"{num_questions} well-structured MCQs with 4 options each, correct answers, and explanations.",
        agent=mcq_agent,
    )
