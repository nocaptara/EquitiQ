# EquitiQ - A News Research Tool

A modern, AI-powered research assistant for news articles. This tool enables users to ask questions in natural language and receive accurate, source-backed answers, leveraging advanced language models, semantic search, and retrieval-augmented generation (RAG).

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Folder Structure](#folder-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [License](#license)

---

## Features

- **Semantic Question Answering:** Ask questions about recent news and get concise, referenced answers.
- **Retrieval-Augmented Generation (RAG):** Combines semantic search with generative AI for robust and transparent answers.
- **Source Attribution:** Every answer includes links to the original news articles for verification.
- **Efficient Document Handling:** Handles large news articles by chunking and indexing for fast retrieval.
- **Scalable & Modular:** Easily add new articles or swap out components (embedding model, LLM, etc.).

---

## Architecture Overview

1. **Document Ingestion:**  
   News articles are loaded from URLs using an unstructured document loader.

2. **Text Chunking:**  
   Articles are split into overlapping chunks (~1000 characters, 200 overlap) to preserve context and enable efficient retrieval.

3. **Embedding Generation:**  
   Each chunk is converted into a semantic vector using HuggingFace's `all-MiniLM-L6-v2` model.

4. **Vector Indexing:**  
   Chunks are indexed using FAISS, a high-performance vector database, for fast similarity search.

5. **LLM Integration:**  
   Google Gemini 1.5 Pro is used via LangChain for answer synthesis, with low temperature for factuality.

6. **RetrievalQA Pipeline:**  
   On user query, relevant chunks are retrieved and passed to the LLM, which generates an answer and cites sources.

7. **Persistence:**  
   The FAISS index is serialized for quick loading and reuse.

---

## Folder Structure

```
.
├── app.py                  # (Optional) Streamlit or UI frontend
├── main.py / notebook.ipynb # Main pipeline and experiments
├── requirements.txt        # Dependencies
├── models/                 # (Optional) Saved models, if any
├── vector_index.pkl        # Serialized FAISS index
├── .env                    # API keys and environment variables
├── README.md               # This file
```

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API Keys

Create a `.env` file with your Google Gemini API key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Download or Prepare News Articles

Update the list of URLs in your loader script or notebook.

### 4. Run the Pipeline

Run the main script or Jupyter notebook to build the vector index and start the QA system.

---

## Usage

- **Ask Questions:**  
  Use the provided interface (script, notebook, or UI) to enter a natural language question about the ingested news articles.

- **Get Answers:**  
  The system returns a concise answer with references to the original articles.

---

## How It Works

1. **Loading Articles:**  
   Loads news articles from provided URLs using `UnstructuredURLLoader`.

2. **Chunking:**  
   Splits each article into overlapping chunks to maintain semantic context.

3. **Embedding:**  
   Converts each chunk into a vector using `sentence-transformers/all-MiniLM-L6-v2`.

4. **Indexing:**  
   Stores all vectors in a FAISS index for efficient similarity search.

5. **Querying:**  
   On a user question, retrieves the top relevant chunks from FAISS.

6. **LLM Answering:**  
   Passes the question and retrieved chunks to Gemini 1.5 Pro, which generates an answer and cites sources.

7. **Output:**  
   Returns the answer and source URLs to the user.

---

## License

This project is for educational and research use only.  
© 2025 Your Name/Organization. All rights reserved.
