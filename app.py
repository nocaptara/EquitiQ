import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import pickle
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredURLLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

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

# Step 1: Get URLs from user
urls = []
print("Enter up to 3 news URLs (press Enter to skip):")
for i in range(3):
    url = input(f"URL {i+1}: ").strip()
    if url:
        urls.append(url)

# Step 2: Process URLs if entered
if urls:
    print("Loading URLs...")
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)

    print("Building vector index with embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

    print("Processing complete and FAISS index saved.")

# Step 3: Query loop
if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)

    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    while True:
        query = input("\nAsk a question based on the articles (or type 'exit'): ").strip()
        if query.lower() == "exit":
            break

        response = qa_chain.invoke({"query": query})
        print("\nAnswer:\n", response["result"])

        print("\nSource Documents:")
        for i, doc in enumerate(response["source_documents"]):
            print(f"\nSource {i+1}: {doc.metadata.get('source', 'URL not found')}")
            print(doc.page_content[:300] + "...")
else:
    print("No FAISS index found. Please enter URLs first.")
