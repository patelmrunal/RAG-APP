from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_text_splitters import CharacterTextSplitter  # Modern 2026 import path

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
sec_key = os.getenv("GROQ_API_KEY")


def load_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    return text


st.title("PDF Chatbot with HuggingFace and LangChain")

uploaded_file = st.file_uploader("upload a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_text = load_pdf(uploaded_file)
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(pdf_text)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    docsearch = FAISS.from_texts(chunks, embeddings)
    llm = ChatGroq(model_name="groq/compound", temperature=0.2, groq_api_key=sec_key)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=docsearch.as_retriever()
    )
    st.success("PDF loaded and system ready")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    question = st.text_input("Ask a question about a PDF!")

    if question:
        result = qa_chain(
            {"question": question, "chat_history": st.session_state.chat_history}
        )
        st.session_state.chat_history.append((question, result["answer"]))
        st.write("**Answer**", result["answer"])
