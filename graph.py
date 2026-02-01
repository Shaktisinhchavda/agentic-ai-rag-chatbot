from typing import TypedDict, List
from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
import os


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ---- Setup ----

emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"}
)

pc = Pinecone()
index = pc.Index("chatbot")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
)

# ---- State ----
class AgentState(TypedDict):
    question: str
    context: List[str]
    scores: List[float]
    answer: str

# ---- Retrieve ----
def retrieve(state: AgentState) -> AgentState:
    query_vec = emb.embed_query(state["question"])
    res = index.query(vector=query_vec, top_k=5, include_metadata=True)

    contexts = [m["metadata"]["text"] for m in res["matches"]]
    scores = [m["score"] for m in res["matches"]]

    return {"context": contexts, "scores": scores}

# ---- Generate ----
def generate(state: AgentState) -> AgentState:
    context_text = "\n\n".join(state["context"])

    prompt = f"""
You are a RAG assistant. Answer ONLY from the provided context.
If the answer is not in the context, say: "Not found in the document."

Context:
{context_text}

Question: {state['question']}
Answer:
"""

    ans = llm.invoke(prompt).content
    return {"answer": ans}

# ---- Graph ----
graph = StateGraph(AgentState)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")

agent= graph.compile()
