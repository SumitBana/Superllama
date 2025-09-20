# SuperLlama (CLI) – RAG Implementation on LlaMa 3.2 1B 

SuperLlama is a **local, privacy-first, document-aware chatbot** that helps you quickly find information inside your files.  
Powered by **Meta LlaMa 3.2 1B**, it runs entirely on your machine – no cloud required, no data leaves your system.

---

## ✨ Features

- 🔒 **Privacy-First:** Runs completely offline, keeps your data secure.  
- 📄 **Smart Document Search:** Quickly find answers in your PDFs, DOCX, or TXT files.  
- 🧠 **Chunk-Based Processing:** Breaks large documents into smaller parts for better context understanding.  
- 🎯 **Semantic Search:** Finds meaning-based answers, not just keyword matches.  
- 🤖 **LLM-Powered Responses:** Synthesizes clear, human-like answers grounded in your documents.

---

## 🖼️ Screenshots

| Welcome Screen |
|---------------|
|<img width="1556" height="789" alt="Screenshot 2025-09-20 222202" src="https://github.com/user-attachments/assets/3e48ad39-54cb-45d4-8529-98a64bfe0a1f" />|

| Project Info |
|---------------|
|<img width="1298" height="825" alt="Screenshot 2025-09-20 222300" src="https://github.com/user-attachments/assets/e96c9a6a-083a-4550-b1e0-8c0e1fcac59a" />|

| Live Demo |
|---------------|
|<img width="1614" height="794" alt="Screenshot 2025-09-20 223624" src="https://github.com/user-attachments/assets/e7256ccf-febf-49ad-9040-320a8f5cfb86" />|

---



## ⚙️ How It Works

1. **Document Chunking**  
   - Breaks large documents into 1024-character chunks.  
   - Adds 200-character overlap to preserve context.

2. **Embedding Generation**  
   - Uses [`sentence-transformers/all-mpnet-base-v2`](https://www.sbert.net/docs/pretrained_models.html)  
   - Converts each chunk into a numerical vector (semantic representation).

3. **Vector Indexing & Retrieval**  
   - Stores vectors in a **VectorStoreIndex** for fast semantic search.  
   - Finds chunks most similar to your query vector.

4. **LLM-Powered Answering**  
   - Sends retrieved chunks + your question to **Meta Llama 3.2 1B**.  
   - Produces a final, human-readable answer grounded in the documents.

---

## 🛠️ Tech Stack

- **LLM:** [Meta Llama 3.2 1B Instruct](https://ai.meta.com/llama/)  
- **Embeddings:** [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)  
- **Vector Index:** [LlamaIndex / FAISS](https://www.llamaindex.ai/)  
- **Interface:** Beautiful TUI (Text User Interface) with Rich/Color formatting  
- **Language:** Python 3.11+

---
