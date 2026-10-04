# InsightPilot

**AI-Powered Document Intelligence using Retrieval-Augmented Generation (RAG)**

InsightPilot is an AI-powered document intelligence application that allows users to upload PDF documents and ask questions using natural language.

The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from the uploaded document and generate grounded responses with source-page citations.

## Features

- PDF document upload and text extraction
- Sentence-aware text chunking
- Semantic search using BGE embeddings
- FAISS vector similarity search
- Retrieval-Augmented Generation (RAG)
- Gemini-powered natural language responses
- Source-page citations
- Hallucination guardrails
- Streamlit web interface
- Whole-document summarization — ongoing enhancement
- Excel/CSV analytics — planned enhancement

## Architecture

PDF Upload  
→ PyMuPDF Text Extraction  
→ Text Cleaning  
→ Sentence-Aware Chunking  
→ BGE Embeddings  
→ FAISS Vector Index  
→ Semantic Retrieval  
→ Gemini  
→ Grounded Answer + Page Citations

## Tech Stack

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- BAAI/bge-small-en-v1.5
- FAISS
- Gemini API
- NLTK
- NumPy

## Project Status

InsightPilot is currently an MVP focused on PDF question answering.

Document summarization and structured-data analytics for Excel/CSV files are planned enhancements.

## Developer

Developed by **Angela Alminanza**
