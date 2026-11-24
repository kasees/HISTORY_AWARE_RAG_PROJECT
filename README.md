# Multimodal RAG Project

A Retrieval-Augmented Generation (RAG) pipeline built with Python and Groq.

## Structure
- `docs/` — Source documents
- `1_ingestion_pipeline.py` — Loads and embeds documents
- `2_retrieval_pipeline.py` — Retrieves relevant context
- `3_history_aware_generation.py` — Generates history-aware answers

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Add your API key to `.env`: `GROQ_API_KEY=your_key_here`
