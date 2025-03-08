import os
print("All Env Variables:", os.environ)

if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("Google API Key is missing! Set it before running the script.")
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env
api_key = os.getenv("GOOGLE_API_KEY")

print(f"API Key Loaded: {api_key}")  # Debugging line

if not api_key:
    raise ValueError("Google API Key is missing! Set it before running the script.")

os.environ["GOOGLE_API_KEY"] = api_key  # Set it in environment


from PyPDF2 import PdfReader

def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_text_chunks(text, chunk_size=1000, overlap=20):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

from langchain.embeddings import GooglePalmEmbeddings

def get_text_embeddings(text_chunks):
    embedding_model = GooglePalmEmbeddings(model_name='models/embedding-001',google_api_key='AIzaSyDrqsCWeZSLqD3M_m-6Ut48N8jg9Yrmnws')
    embeddings = embedding_model.embed_documents(text_chunks)
    return embeddings


import faiss
import numpy as np

def store_embeddings(embeddings):
    dim = len(embeddings[0])  # DimensionAIzaSyCKH3Bd25YeqA1WDqs1KglfgeYOQP_MY3s of embeddings
    index = faiss.IndexFlatL2(dim)  # Create FAISS index
    index.add(np.array(embeddings))  # Store embeddings
    return index

from langchain.llms import GooglePalm
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

def get_conversational_chain(vector_store):
    llm = GooglePalm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain(llm=llm, retriever=vector_store, memory=memory)

import streamlit as st

def main():
    st.set_page_config(page_title="DocuQuery: AI-Powered PDF Knowledge Assistant")
    st.header("DocuQuery: AI-Powered PDF Assistant")

    uploaded_files = st.sidebar.file_uploader("Upload PDFs", accept_multiple_files=True)
    process_button = st.sidebar.button("Process")

    if process_button and uploaded_files:
        with st.spinner("Processing..."):
            text = get_pdf_text(uploaded_files)
            chunks = get_text_chunks(text)
            embeddings = get_text_embeddings(chunks)
            vector_store = store_embeddings(embeddings)
            conversation_chain = get_conversational_chain(vector_store)
        st.success("Processing complete! Ask questions below.")
        
        user_question = st.text_input("Ask a question:")
        if user_question:
            response = conversation_chain.run(user_question)
            st.write("Bot:", response)

if __name__ == "__main__":
    main()


