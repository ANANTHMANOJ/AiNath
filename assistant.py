# Importing required libraries
import pandas as pd
import numpy as np
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate  
from langchain_community.document_loaders import PyPDFDirectoryLoader  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Loading env variables
os.environ["GROQ_API_KEY"]  = st.secrets["GROQ_API_KEY"]

# Initializing Groq llm with qwen-2.5-32b  model
llm = ChatGroq(model="qwen-2.5-32b",verbose=True,)

# Tuning the llm with the predifine prompt
prompt = ChatPromptTemplate.from_template("""
Assume you are an Assistant of the person in the context.
Please provide the most accurate answer based on the questions in formal and attractive way. Act like an Assistant acts like. 
Answer the questions about that person based on the context only
If asked Who are you, answer like you are assistant of Ananthmanoj.
<context>
{context}
<context>
Question: {input}
""")

# Function to create the embedding using Ollama Embedding after splitting the document into chunks
def create_vector_embeddings():
    if 'vector' not in st.session_state:
        st.session_state.loader = PyPDFDirectoryLoader("docs")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.embeddings = OllamaEmbeddings(model="llama3.2:1b")
        st.session_state.vectorstore = FAISS.from_documents(documents=st.session_state.final_docs, embedding=st.session_state.embeddings)
        print("Done")

# Using streamlit as UI and to take input and display output       
st.set_page_config(page_title="Know About Ananthmanoj")
st.title("Ananthmanoj Unplugged – Data & Discovery 🏴‍☠️📊")


try:
    with st.spinner("Getting ready..."):
        create_vector_embeddings()
        st.write("Connected")
    
    user_prompt = st.text_input("Hi, I am AiNath, What question do you have about my Master ?")
    if user_prompt:
        with st.spinner("He has many good things...let me summarise",show_time= True):
            doc_chain = create_stuff_documents_chain(llm=llm, prompt= prompt)
            retriever = st.session_state.vectorstore.as_retriever()
            rertriever_chain = create_retrieval_chain(retriever, doc_chain)
            response = rertriever_chain.invoke({"input":user_prompt})
            st.success(response['answer'])

        with st.expander("Similar results :"):
            for i,d in enumerate(response["context"]):
                st.write(d.page_content)
                st.write("------------------------------------")

    
except Exception as e:
    st.write(f"ERROR!! Contact My Master \n error: ({e})")
