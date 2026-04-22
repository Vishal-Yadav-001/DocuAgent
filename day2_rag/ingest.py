"""
Day 2 - Script 1: Document Ingestion Into Vector Store
Chunks PDF and CSV data, embeds them, and stores in ChromaDB.
Supports large PDFs with mixed text + images using PyMuPDF.
Run: python day2_rag/ingest.py
"""

import os
import sys
import shutil
import pandas as pd
import fitz  # pymupdf - better than pdfplumber for complex/large PDFs
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")
CSV_DIR = os.path.join(BASE_DIR, "data", "csvs")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")


def load_pdf_text(path: str) -> Document:
    """
    Extract ALL text from a PDF as one continuous string (pageless).
    Concatenating pages before chunking keeps tables and context intact -
    financial table headers and their numbers end up in the same chunk.
    """
    filename = os.path.basename(path)
    print(f"  Reading: {filename}")
    try:
        pdf = fitz.open(path)
    except Exception as e:
        print(f"  [ERROR] Could not open {filename}: {e}")
        return None

    total_pages = len(pdf)
    all_text = []
    skipped = 0

    for i, page in enumerate(pdf):
        try:
            text = page.get_text("text").strip()
        except Exception:
            skipped += 1
            continue
        if not text:
            skipped += 1
            continue
        all_text.append(text)
        if (i + 1) % 50 == 0:
            print(f"    ... processed {i + 1}/{total_pages} pages")

    pdf.close()
    full_text = "\n\n".join(all_text)
    if not full_text.strip():
        print(f"  [WARNING] No text extracted from {filename} (image-only PDF?)")
        return None
    print(
        f"  Extracted {len(full_text):,} chars from {total_pages - skipped}/{total_pages} pages "
        f"({skipped} image-only skipped)"
    )
    return Document(
        page_content=full_text,
        metadata={"source": filename}
    )


def load_all_pdfs(pdf_dir: str) -> list[Document]:
    """Load all PDF files found in the given directory."""
    all_docs = []
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"[WARNING] No PDF files found in {pdf_dir}")
        return all_docs

    print(f"Found {len(pdf_files)} PDF file(s): {', '.join(pdf_files)}")
    for pdf_file in pdf_files:
        path = os.path.join(pdf_dir, pdf_file)
        doc = load_pdf_text(path)
        if doc is not None:
            all_docs.append(doc)

    return all_docs


def load_all_csvs(csv_dir: str) -> list[Document]:
    """Load all CSV files found in the given directory."""
    all_docs = []
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]

    if not csv_files:
        print(f"[WARNING] No CSV files found in {csv_dir}")
        return all_docs

    print(f"Found {len(csv_files)} CSV file(s): {', '.join(csv_files)}")
    for csv_file in csv_files:
        path = os.path.join(csv_dir, csv_file)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  [ERROR] Could not read {csv_file}: {e}")
            continue
        lines = [f"CSV File: {csv_file}", df.to_string(index=False), "\nStatistics:", df.describe().to_string()]
        all_docs.append(Document(
            page_content="\n".join(lines),
            metadata={"source": csv_file, "type": "csv"}
        ))
        print(f"  Loaded: {csv_file} ({len(df)} rows)")

    return all_docs


def ingest_documents():
    """Main ingestion pipeline: load -> chunk -> embed -> store in ChromaDB."""
    print("=" * 60)
    print("STEP 1: Loading documents")
    print("=" * 60)
    pdf_docs = load_all_pdfs(PDF_DIR)
    csv_docs = load_all_csvs(CSV_DIR)

    all_docs = pdf_docs + csv_docs
    if not all_docs:
        print("[ERROR] No documents loaded. Add PDF or CSV files to data/ and retry.")
        sys.exit(1)
    
    return create_vectorstore(all_docs)


def create_vectorstore(documents: list[Document], clear_old: bool = True):
    """Chunk documents, embed them, and store in ChromaDB."""
    total_chars = sum(len(d.page_content) for d in documents)
    print(f"\nTotal: {len(documents)} pages/documents, {total_chars:,} characters")

    print("\n" + "=" * 60)
    print("STEP 2: Chunking text")
    print("=" * 60)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} pages")

    print("\n" + "=" * 60)
    print("STEP 3: Loading embedding model")
    print("=" * 60)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    print("\n" + "=" * 60)
    print("STEP 4: Storing in ChromaDB")
    print("=" * 60)
    if clear_old and os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Cleared old ChromaDB")

    batch_size = 100
    print(f"Embedding and storing {len(chunks)} chunks in batches of {batch_size}...")

    vectorstore = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        if vectorstore is None and (clear_old or not os.path.exists(CHROMA_PATH)):
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=CHROMA_PATH,
            )
        else:
            if vectorstore is None:
                vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
            vectorstore.add_documents(batch)
        print(
            f"  Stored batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1} "
            f"({min(i + batch_size, len(chunks))}/{len(chunks)} chunks)"
        )

    print(f"\n[SUCCESS] Ingested {len(chunks)} chunks into ChromaDB at: {CHROMA_PATH}")
    return vectorstore


if __name__ == "__main__":
    ingest_documents()
