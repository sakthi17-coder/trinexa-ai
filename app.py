import streamlit as st
import os
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. UI Configuration
st.set_page_config(page_title="TRINEXA AI", layout="wide")
st.title("TRINEXA | Company Intelligence Hub")

# 2. Secure API Connection (Using Streamlit Secrets)
if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["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"])
else:
    st.error("Missing GROQ_API_KEY! Add it to App Settings > Secrets.")
    st.stop()

# 3. Persistent Memory
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# 4. Sidebar: PDF Knowledge Upload
with st.sidebar:
    st.header("Upload Knowledge")
    uploaded_file = st.file_uploader("Upload Company PDF", type="pdf")
    
    if uploaded_file and st.button("Train TRINEXA"):
        with st.status("Reading document..."):
            # Save temporary file for LangChain loader
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load and chunk the PDF
            loader = PyPDFLoader("temp.pdf")
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(pages)
            
            # Build Vector Database using free CPU embeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.vector_db = FAISS.from_documents(docs, embeddings)
            st.success("TRINEXA is now trained on this document!")

# 5. Professional Chat Interface
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask TRINEXA..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Retrieval: Pull context from the PDF if it exists
    context = ""
    if st.session_state.vector_db:
        search_results = st.session_state.vector_db.similarity_search(prompt, k=3)
        context = "\n".join([doc.page_content for doc in search_results])

    # Generation: DeepSeek-R1 Reasoning
    with st.chat_message("assistant"):
        system_prompt = f"Context: {context}\nYou are TRINEXA, a reasoning AI. Use the provided context to answer accurately."
        
        completion = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": prompt}],
            stream=True
        )
        
        response = st.write_stream(completion)
    st.session_state.messages.append({"role": "assistant", "content": response})
