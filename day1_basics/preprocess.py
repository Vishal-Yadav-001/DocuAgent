"""
Day 1 - Script 1: Data Preprocessing
Loads and cleans PDF and CSV files, prints extracted content.
Run: python day1_basics/preprocess.py
"""

import os
import sys
import pandas as pd
import pdfplumber

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(BASE_DIR, "data", "pdfs", "q1fy26-earnings-presentation.pdf")
CSV_PATH = os.path.join(BASE_DIR, "data", "csvs", "financial_data.csv")


def load_pdf(path: str) -> str:
    """Extract all text from a PDF file."""
    if not os.path.exists(path):
        print(f"[ERROR] PDF not found at {path}")
        print("Add a PDF file to data/pdfs/ first.")
        sys.exit(1)

    text = ""
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {i + 1} ---\n{page_text}"
    return text.strip()


def load_csv(path: str) -> pd.DataFrame:
    """Load and display basic info about a CSV file."""
    if not os.path.exists(path):
        print(f"[ERROR] CSV not found at {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    return df


def summarize_dataframe(df: pd.DataFrame) -> str:
    """Generate a text summary of a dataframe for LLM input."""
    summary = []
    summary.append(f"Dataset has {len(df)} rows and {len(df.columns)} columns.")
    summary.append(f"Columns: {', '.join(df.columns.tolist())}")
    summary.append("\nNumerical Summary:")
    summary.append(df.describe().to_string())
    summary.append("\nSample Data (first 3 rows):")
    summary.append(df.head(3).to_string(index=False))
    return "\n".join(summary)


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1: Loading PDF")
    print("=" * 60)
    pdf_text = load_pdf(PDF_PATH)
    print(f"Extracted {len(pdf_text)} characters from PDF.")
    print("\nFirst 500 characters preview:")
    print(pdf_text[:500])

    print("\n" + "=" * 60)
    print("STEP 2: Loading CSV")
    print("=" * 60)
    df = load_csv(CSV_PATH)
    print(summarize_dataframe(df))

    print("\n[SUCCESS] Data preprocessing complete!")
