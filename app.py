import streamlit as st
from graph import agent

st.set_page_config(page_title="Agentic AI RAG Chatbot")

st.title("Agentic AI eBook Chatbot")

question = st.text_input("Ask a question from the eBook")

if question:
    result = agent.invoke({"question": question})

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Retrieved Context")
    for i, j in enumerate(result["context"]):
        st.markdown(f"Chunk {i+1} (score: {result['scores'][i]:.3f})")
        st.write(j)
