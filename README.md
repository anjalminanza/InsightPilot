# InsightPilot

**AI-Powered Document Intelligence using Retrieval-Augmented Generation (RAG)**

InsightPilot is a deployed AI-powered document intelligence application that enables users to upload PDF documents and ask questions using natural language.

The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from uploaded documents and generate grounded responses with source-page citations.

🔗 **Live Demo:** https://insightpilot-jcrckdcpifct2mld5vs65v.streamlit.app/

## Features

- PDF upload and text extraction
- Sentence-aware document chunking
- Semantic search using BGE embeddings
- FAISS vector similarity search
- Retrieval-Augmented Generation (RAG)
- Gemini-powered natural language responses
- Source-page citations
- Hallucination guardrails
- Deployed Streamlit web application
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

The current MVP supports end-to-end PDF question answering using RAG, including semantic retrieval, grounded response generation, and source-page citations.

Whole-document summarization and structured-data analytics for Excel/CSV files are planned enhancements.

## Developer

Developed by **Angela Alminanza**
