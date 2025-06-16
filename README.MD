# 🔍 RAG PDF Assistant

A **Retrieval-Augmented Generation (RAG)** based PDF Assistant that allows users to:
- Upload a research paper or PDF,
- Automatically convert it to vector embeddings using `FAISS`,
- Ask questions in natural language,
- Get accurate, context-aware answers powered by **HuggingFace models**.

Built using:
- 🤗 HuggingFace Transformers + LangChain
- 🧠 FAISS for Vector DB
- 📄 PyMuPDF (`fitz`) for PDF parsing
- 🖼️ Gradio for UI



---


## 🛠️ Features

✅ Upload and process any academic or technical PDF  
✅ Automatically splits text and stores embeddings using `FAISS`  
✅ Asks questions in natural language (RAG-based answers)  
✅ Clean Gradio UI  
✅ Uses `google/flan-t5-base` for lightweight local inference  

---

## 📂 Project Structure

```bash
├── data/
│   └── your_uploaded.pdf #initially, to be used for testing purposes before the interface development
├── vectorstore/
│   └── FAISS index
├── rag.py              # Core logic for PDF parsing, embedding, and QA
├── app.py              # Gradio UI app
├── requirements.txt    # Dependencies
└── README.md           # This file
```

<pre lang="markdown"> ## 🧠 Architecture ```mermaid graph TD A[PDF Upload] --> B[Extract Text with PyMuPDF] B --> C[Split Text into Chunks] C --> D[Convert Chunks to Embeddings via MiniLM] D --> E[Store in FAISS VectorDB] F[User Question] --> G[Retrieve Relevant Chunks] G --> H[FLAN-T5 Inference via LangChain] H --> I[Answer Returned] ``` </pre>