from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ---- Embeddings 
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"}
)

# ---- Pinecone Index ----
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "chatbot"

index = pc.Index(index_name)

# ---- Load PDF ----
loader = PyPDFLoader("data/Ebook-Agentic-AI.pdf")
docs = loader.load()

# ---- Chunking ----
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
chunks = splitter.split_documents(docs)

# ---- Upsert ----
vectors = []
for i, chunk in enumerate(chunks):
    vec = embeddings.embed_query(chunk.page_content)
    vectors.append((
        f"id-{i}",
        vec,
        {"text": chunk.page_content}
    ))

index.upsert(vectors=vectors)
print("Ingestion completed")
