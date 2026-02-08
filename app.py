import streamlit as st
import os
import tempfile

from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory

from core.processor import load_and_split_pdf
from core.vector_store import create_and_store_faiss, load_faiss_from_s3
from core.chat_engine import get_session_history
from core.rag_chain import build_rag_chain
from utils.s3_utils import list_docs

# CONFIG
S3_BUCKET = "your-bucket-name"
BASE_PREFIX = "faiss_indexes/"

st.title("🚀 Multi-Document FAISS RAG")

api_key = st.text_input("Groq API Key", type="password")
session_id = st.text_input("Session ID", value="default")

if "store" not in st.session_state:
    st.session_state.store = {}

existing_docs = list_docs(S3_BUCKET, BASE_PREFIX)
selected_docs = st.multiselect("Select documents", existing_docs)

uploaded_file = st.file_uploader("Upload new PDF", type="pdf")

if uploaded_file:
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = os.path.join(tmp, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        splits = load_and_split_pdf(pdf_path)
        create_and_store_faiss(
            splits,
            S3_BUCKET,
            uploaded_file.name.replace(".pdf", ""),
            BASE_PREFIX
        )

        st.success("Document indexed and saved!")

if selected_docs and api_key:
    vectorstore = load_faiss_from_s3(
        S3_BUCKET,
        selected_docs,
        BASE_PREFIX
    )

    retriever = vectorstore.as_retriever()
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    rag_chain = build_rag_chain(llm, retriever)

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda s: get_session_history(s, st.session_state.store),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    query = st.text_input("Ask a question")
    if query:
        response = conversational_chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}}
        )
        st.write("🧠 Answer:", response["answer"])
