# PDF RAG Chatbot

A simple Retrieval-Augmented Generation (RAG) project built with Python and LangChain. It lets you ask questions about a PDF using local embedding and language models.

## Features

- Load PDF documents
- Split text into chunks
- Generate embeddings using Ollama (`nomic-embed-text`)
- Store vectors in ChromaDB
- Reuse saved embeddings on future runs
- Retrieve relevant context
- Answer questions using Qwen 2.5 running locally

## Tech Stack

- Python
- LangChain
- Ollama
- ChromaDB
- Qwen 2.5
- nomic-embed-text

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Pull the required models:

```bash
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text
```

Run:

```bash
python app.py
```

## What I Learned

- Building a complete RAG pipeline
- Working with vector databases
- Semantic search using embeddings
- Local LLM inference with Ollama
- LangChain retrieval chains
