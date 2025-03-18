# app.py
import os
import pickle
import streamlit as st
from dotenv import load_dotenv
import warnings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
file_path = "faiss_store_gemini.pkl"

# Initialize Gemini model via LangChain
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-pro-latest",
    google_api_key=api_key,
    temperature=0.1,
    convert_system_message_to_human=True,
)

# Streamlit UI
st.set_page_config(page_title="News Research Tool", layout="centered")
st.title("📰 News Research Tool")
st.markdown("Enter up to 3 news article URLs and ask questions about them.")

# URL Input Fields
urls = []
for i in range(3):
    url = st.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

if st.button("Process URLs"):
    if not urls:
        st.warning("Please enter at least one URL.")
    else:
        with st.spinner("Loading and processing URLs..."):
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(data)

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(docs, embeddings)

            with open(file_path, "wb") as f:
                pickle.dump(vectorstore, f)

        st.success("✅ Processing complete and FAISS index saved!")

# Query Input Section
if os.path.exists(file_path):
    query = st.text_input("🔍 Ask a question based on the articles:")
    if query:
        with st.spinner("Generating answer..."):
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)

            retriever = vectorstore.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True
            )

            response = qa_chain.invoke({"query": query})
            st.subheader("📌 Answer")
            st.write(response["result"])

            st.subheader("📂 Source Documents")
            for i, doc in enumerate(response["source_documents"]):
                st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'URL not found')}")
                st.write(doc.page_content[:300] + "...")
else:
    st.info("Please process URLs first before asking questions.")
