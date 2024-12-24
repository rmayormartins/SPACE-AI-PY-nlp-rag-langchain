---
title:  Multilingual RAG Question-Answering System
emoji: 🈁↔️🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.7.1
app_file: app.py
pinned: false
license: ecl-2.0
---

# Multilingual RAG Question-Answering System

This project implements a Retrieval-Augmented Generation (RAG) system for question answering in multiple languages. It uses advanced language models and embeddings to provide accurate answers based on provided texts.

## Developer
Developed by Ramon Mayor Martins (2024)
* Email: rmayormartins@gmail.com
* Homepage: https://rmayormartins.github.io/
* Twitter: @rmayormartins
* GitHub: https://github.com/rmayormartins
* Space: https://huggingface.co/rmayormartins

## Technologies Used

* **LangChain:** Framework for developing applications powered by language models, providing tools for document loading, text splitting, and creating chains of operations.

* **Sentence Transformers:** Library for state-of-the-art text embeddings, using the multilingual-e5-large model for superior multilingual understanding.

* **Flan-T5:** Advanced language model from Google that excels at various NLP tasks, particularly strong in multilingual text generation and understanding.

* **Chroma DB:** Lightweight vector database for storing and retrieving text embeddings efficiently, enabling semantic search capabilities.

* **Gradio:** Framework for creating user-friendly web interfaces for machine learning models, providing an intuitive way to interact with the RAG system.

* **HuggingFace Transformers:** Library providing access to state-of-the-art transformer models, tokenizers, and pipelines.

* **PyTorch:** Deep learning framework that powers the underlying models and computations.

## Key Features

* **Multilingual Support:** Process and answer questions in multiple languages (English, Spanish, Portuguese, and more)
* **Document Chunking:** Smart text splitting for handling long documents
* **Semantic Search:** Uses advanced embeddings for accurate information retrieval
* **Source Attribution:** Provides references to the relevant text passages used for answers
* **User-Friendly Interface:** Simple web interface for text input and question answering

## How it Works

1. **Text Processing:**
   - User inputs a text document
   - System splits text into manageable chunks
   - Chunks are converted into embeddings using multilingual-e5-large

2. **Knowledge Base Creation:**
   - Embeddings are stored in Chroma vector database
   - Document metadata is preserved for source attribution

3. **Question Answering:**
   - User asks a question in any supported language
   - System retrieves relevant document chunks
   - Flan-T5 generates a coherent answer based on retrieved context
   - Sources are displayed for transparency

## How to Use

1. Open the application interface
2. Paste your reference text in the "Base Text" field
3. Enter your question in any supported language
4. Receive an answer along with relevant source excerpts

## Example Use Cases

* Document analysis and comprehension
* Educational Q&A systems
* Multilingual information retrieval
* Research assistance
* Content summarization

## Technical Architecture

* **Embedding Model:** intfloat/multilingual-e5-large
* **Language Model:** google/flan-t5-large
* **Vector Store:** Chroma
* **Chunk Size:** 500 characters
* **Context Window:** 4 documents

## Local Development

```bash
pip install -r requirements.txt
python app.py
```

## Deployment

This application is deployed on Hugging Face Spaces. You can access it at [https://huggingface.co/spaces/rmayormartins/nlp-rag-langchain].

## Note

The system's responses are generated solely based on the provided text. The quality of answers depends on the content and clarity of the input text.