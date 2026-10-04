
# ============================================================
# INSIGHTPILOT - RAG BACKEND
# ============================================================

import re
import time
from typing import Any, Dict, List

import faiss
import nltk
import numpy as np
import pymupdf

from google.genai import errors
from nltk.tokenize import sent_tokenize


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
GEMINI_MODEL_NAME = "gemini-3.8-flash"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 5

FALLBACK_MESSAGE = (
    "I could not find enough information in the document."
)

SUMMARY_FEATURE_MESSAGE = (
    "Whole-document summarization is an ongoing enhancement "
    "and will be available in a future version of InsightPilot."
)

nltk.download("punkt_tab", quiet=True)


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_pages(file_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extract text from every page of an uploaded PDF.

    Streamlit provides the uploaded PDF as bytes, so PyMuPDF
    opens the document directly from memory.
    """

    document = pymupdf.open(
        stream=file_bytes,
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(document, start=1):

        pages.append({
            "page": page_number,
            "text": page.get_text("text")
        })

    document.close()

    return pages


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_pdf_text(text: str) -> str:
    """Clean common PDF text-extraction artifacts."""

    # Remove control characters.
    text = re.sub(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]",
        " ",
        text
    )

    # Normalize repeated whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_pdf_pages(
    pages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Clean extracted text while preserving page numbers."""

    cleaned_pages = []

    for page in pages:

        cleaned_text = clean_pdf_text(
            page["text"]
        )

        if cleaned_text:

            cleaned_pages.append({
                "page": page["page"],
                "text": cleaned_text
            })

    return cleaned_pages


# ============================================================
# SENTENCE-AWARE CHUNKING
# ============================================================

def split_text_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """
    Split text into chunks while trying to preserve
    sentence boundaries.
    """

    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        candidate = (
            f"{current_chunk} {sentence}".strip()
            if current_chunk
            else sentence
        )

        if len(candidate) <= chunk_size:

            current_chunk = candidate

        else:

            if current_chunk:
                chunks.append(current_chunk)

            # Keep a small amount of previous context.
            overlap_text = (
                current_chunk[-overlap:]
                if current_chunk
                else ""
            )

            current_chunk = (
                f"{overlap_text} {sentence}".strip()
                if overlap_text
                else sentence
            )

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def create_pdf_chunks(
    pages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Create searchable chunks while retaining
    the source page for citations.
    """

    chunks = []

    for page in pages:

        page_chunks = split_text_into_chunks(
            page["text"]
        )

        for chunk in page_chunks:

            chunks.append({
                "page": page["page"],
                "text": chunk
            })

    return chunks


# ============================================================
# EMBEDDINGS
# ============================================================

def create_chunk_embeddings(
    chunks: List[Dict[str, Any]],
    embedding_model
) -> np.ndarray:
    """Generate normalized BGE embeddings."""

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return np.asarray(
        embeddings,
        dtype="float32"
    )


# ============================================================
# FAISS VECTOR INDEX
# ============================================================

def create_faiss_index(
    embeddings: np.ndarray
) -> faiss.IndexFlatIP:
    """
    Create a FAISS inner-product index.

    Since the embeddings are normalized,
    inner product behaves like cosine similarity.
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# ============================================================
# COMPLETE PDF PROCESSING PIPELINE
# ============================================================

def process_pdf(
    file_bytes: bytes,
    embedding_model
) -> Dict[str, Any]:
    """
    Process an uploaded PDF from extraction
    through vector indexing.
    """

    pages = extract_pdf_pages(
        file_bytes
    )

    cleaned_pages = clean_pdf_pages(
        pages
    )

    chunks = create_pdf_chunks(
        cleaned_pages
    )

    if not chunks:

        raise ValueError(
            "No readable text was found in the PDF."
        )

    embeddings = create_chunk_embeddings(
        chunks,
        embedding_model
    )

    index = create_faiss_index(
        embeddings
    )

    return {
        "chunks": chunks,
        "index": index,
        "page_count": len(pages),
        "chunk_count": len(chunks)
    }


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def search_pdf(
    question: str,
    chunks: List[Dict[str, Any]],
    index,
    embedding_model,
    top_k: int = DEFAULT_TOP_K
) -> List[Dict[str, Any]]:
    """
    Retrieve the PDF chunks most relevant
    to the user's question.
    """

    # BGE retrieval works better with this
    # query instruction.
    query = (
        "Represent this sentence for searching "
        "relevant passages: "
        + question
    )

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for score, chunk_index in zip(
        scores[0],
        indices[0]
    ):

        if chunk_index < 0:
            continue

        chunk = chunks[
            chunk_index
        ]

        results.append({
            "page": chunk["page"],
            "text": chunk["text"],
            "score": float(score)
        })

    return results


# ============================================================
# RAG PROMPT
# ============================================================

def build_rag_prompt(
    question: str,
    retrieved_chunks: List[Dict[str, Any]]
) -> str:
    """
    Build a grounded Gemini prompt using only
    the retrieved document context.
    """

    context = "\n\n".join(
        f"[Page {item['page']}]\n{item['text']}"
        for item in retrieved_chunks
    )

    prompt = f"""
You are InsightPilot, an AI document intelligence assistant.

Answer the user's question using ONLY the document context below.

Rules:
- Do not use outside knowledge.
- If the answer cannot be found in the context, respond exactly:
  "{FALLBACK_MESSAGE}"
- Give a clear and concise answer.
- Do not invent information.
- Cite relevant page numbers using [Page X].

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    return prompt.strip()


# ============================================================
# GEMINI
# ============================================================

def call_gemini(
    prompt: str,
    client,
    max_retries: int = 3
) -> str:
    """
    Send the grounded RAG prompt to Gemini.

    Temporary server errors are retried using
    exponential backoff.
    """

    for attempt in range(
        max_retries
    ):

        try:

            response = (
                client.models.generate_content(
                    model=GEMINI_MODEL_NAME,
                    contents=prompt
                )
            )

            return response.text

        except errors.ServerError:

            if attempt == max_retries - 1:
                raise

            time.sleep(
                2 ** attempt
            )


# ============================================================
# END-TO-END PDF QUESTION ANSWERING
# ============================================================

def answer_pdf_question(
    question: str,
    chunks: List[Dict[str, Any]],
    index,
    embedding_model,
    client
) -> Dict[str, Any]:
    """
    Retrieve relevant document context and
    generate a grounded answer using Gemini.
    """

    retrieved_chunks = search_pdf(
        question=question,
        chunks=chunks,
        index=index,
        embedding_model=embedding_model
    )

    prompt = build_rag_prompt(
        question,
        retrieved_chunks
    )

    answer = call_gemini(
        prompt,
        client
    )

    # If InsightPilot could not answer the question,
    # don't display irrelevant retrieved pages.
    if FALLBACK_MESSAGE in answer:

        source_pages = []

    else:

        # Only display pages actually cited
        # in Gemini's final answer.
        cited_pages = re.findall(
            r"\[Page (\d+)\]",
            answer
        )

        source_pages = sorted(
            set(
                int(page)
                for page in cited_pages
            )
        )

    return {
        "answer": answer,
        "source_pages": source_pages,
        "retrieved_chunks": retrieved_chunks
    }
