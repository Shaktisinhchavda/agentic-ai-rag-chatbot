
#  Agentic AI RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built using **LangGraph**, **Pinecone**, **BGE Embeddings**, **Groq LLM**, and a **Streamlit UI**.  
The chatbot answers questions **strictly grounded** in a provided PDF knowledge base: `Ebook-Agentic-AI.pdf`.

---

##  Features

- PDF → Chunk → Embed → Store in Pinecone
- LangGraph-powered RAG pipeline
- Strict grounding (no hallucinations)
- Returns:
  - Final Answer
  - Retrieved Context Chunks
  - Similarity Scores
- Interactive Streamlit chat UI

---

##  Architecture Overview

```
                ┌──────────────────────┐
                │   Ebook-Agentic-AI  │
                │        (PDF)        │
                └──────────┬──────────┘
                           │
                     Text Chunking
                           │
                    BGE Embeddings
                           │
                       Pinecone
                           │
User Question → Embedding → Vector Search
                           │
                      LangGraph Flow
                           │
                        Groq LLM
                           │
                Answer + Context + Score
                           │
                      Streamlit UI
```

---

##  Tech Stack

| Component | Tool |
|---|---|
| LLM | Groq (`openai/gpt-oss-120b`) |
| Embeddings | `BAAI/bge-large-en-v1.5` |
| Vector DB | Pinecone (cosine, 1024-dim) |
| Orchestration | LangGraph |
| UI | Streamlit |
| PDF Loader | PyPDFLoader |

---

##  Project Structure

```
RAG_Chatbot/
│
├── ingestion.py          # PDF → Pinecone
├── rag_graph.py          # LangGraph RAG pipeline
├── app.py                # Streamlit UI
├── .env                  # API keys
└── Ebook-Agentic-AI.pdf
```

---

##  Setup Instructions

###  Create Virtual Environment

```bash
python -m venv rag_env
rag_env\Scripts\activate   # Windows
```

###  Install Dependencies

```bash
pip install requirements.txt
```

###  Add API Keys

Create a `.env` file:

```
PINECONE_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

###  Place the PDF

Put the knowledge base here:

```
data/Ebook-Agentic-AI.pdf
```

---

##  Step 1 — Ingest PDF into Pinecone

```bash
python ingestion.py
```

---

##  Step 2 — Run the Chatbot UI

```bash
streamlit run app.py
```

---

##  Sample Queries to Test

1. What is Agentic AI?
2. How does Agentic AI differ from earlier AI systems?
3. What are the foundational elements required for Agentic AI adoption?
4. Explain the role of verification in Agentic AI.
5. What challenges do organizations face when implementing Agentic AI?
6. How do AI agents learn over time?

---

##  Grounded Answering (No Hallucination)

If the answer is not present in the retrieved context, the system replies:

**"Not found in the document."**

---

##  Output Format

For every query, the system returns:

-  Final Answer
-  Retrieved Context Chunks
-  Similarity Scores from Pinecone


