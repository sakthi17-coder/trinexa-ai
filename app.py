import streamlit as st
import os
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Professional Page Setup
st.set_page_config(page_title="TRINEXA AI | Company Hub", layout="wide")
st.title("TRINEXA AI - Internal Knowledge Engine")
st.sidebar.title("Company Documents")

# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# 2. Get API Key from Secrets
try:
    api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=api_key)
except:
    st.error("Missing GROQ_API_KEY in Secrets!")
    st.stop()

# 3. PDF Uploader Section (In Sidebar)
uploaded_file = st.sidebar.file_uploader("Upload Company PDF", type="pdf")

if uploaded_file and st.sidebar.button("Process Document"):
    with st.status("Reading document and building brain..."):
        # Save temp file
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and Split PDF into chunks
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(pages)
        
        # Create Vector Database (This is the "Searchable Memory")
        # We use a free embedding model that runs on Hugging Face CPU
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vector_db = FAISS.from_documents(docs, embeddings)
        st.sidebar.success("TRINEXA now knows this document!")

# 4. Chat Interface
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about company policies or generic questions..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 5. The "RAG" Logic (Searching the PDF)
    context = ""
    if st.session_state.vector_db:
        # Search the PDF for the 3 most relevant paragraphs
        search_results = st.session_state.vector_db.similarity_search(prompt, k=3)
        context = "\n".join([doc.page_content for doc in search_results])

    # 6. Generate Response using DeepSeek-R1 (Better than ChatGPT Free)
    with st.chat_message("assistant"):
        full_prompt = f"""
        You are TRINEXA. Use the following internal company context to answer the user. 
        If the answer is not in the context, use your general intelligence to help.
        
        CONTEXT: {context}
        USER QUESTION: {prompt}
        """
        
        completion = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": full_prompt}],
            stream=True
        )
        
        response = st.write_stream(completion)
    st.session_state.messages.append({"role": "assistant", "content": response})