# Importing required libraries
import pandas as pd
import numpy as np
import os
import torch
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate  
from langchain_community.document_loaders import PyPDFDirectoryLoader  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

torch.classes.__path__ = []
os.environ["GROQ_API_KEY"]  = st.secrets["GROQ_API_KEY"]
os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]

# Initialize Groq LLM
llm = ChatGroq(model="compound-beta", verbose=True)

# Define prompt template
prompt = ChatPromptTemplate.from_template("""
You are the personal assistant of Ananthmanoj. Your role is to answer questions about him based solely on the provided context.
Please respond in a professional, articulate, and engaging manner—just as a real assistant would.
If asked “Who are you?”, respond that you are the personal assistant of Ananthmanoj.
If a question is asked and the answer is not available in the context, simply reply with: exploded brain emoji saying "My master didn’t reveal it."
Do not use phrases like “according to the context.”
Always stay in character and ensure your responses reflect loyalty, clarity, and charm.
Let your tone reflect both intelligence and elegance, as an ideal assistant would.
<context>
{context}
<context>
Question: {input}
""")

# Function to create vector embeddings
def create_vector_embeddings():
    if 'vector' not in st.session_state:
        st.session_state.loader = PyPDFDirectoryLoader("docs")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="impira/layoutlm-document-qa")
        st.session_state.vectorstore = FAISS.from_documents(
            documents=st.session_state.final_docs, 
            embedding=st.session_state.embeddings
        )
        print("Vector store created!")

# Set page config
st.set_page_config(page_title="Know About Ananthmanoj", page_icon="🧠", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
        .big-title {
            font-size: 40px;
            color: #3B82F6;
            font-weight: bold;
            text-align: center;
        }
        .small-subtitle {
            text-align: center;
            font-size: 16px;
            color: #6B7280;
        }
        .footer {
            text-align: center;
            font-size: 13px;
            color: #9CA3AF;
            margin-top: 40px;
        }
    </style>
""", unsafe_allow_html=True)

# Title & subtitle
st.markdown('<div class="big-title">Ananthmanoj Unplugged – Data & Discovery 🏴‍☠️📊</div>', unsafe_allow_html=True)
st.markdown('<div class="small-subtitle">Curious about the man behind the magic? Ask me anything!</div>', unsafe_allow_html=True)
st.write("")

try:
    with st.spinner("Initializing Memory & Intelligence... ⚙️🧠"):
        create_vector_embeddings()
        st.success("🔌 Connection with Ananthmanoj’s brain established!")

    # Input area
    user_prompt = st.text_input("🤖 Hi, I am AiNath, what would you like to know about my Master?", placeholder="Type your question here...")

    if user_prompt:
        with st.spinner("Processing your request... 📚"):
            doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
            retriever = st.session_state.vectorstore.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, doc_chain)
            response = retrieval_chain.invoke({"input": user_prompt})
            st.success("🎯 Here's what I found:")
            st.markdown(f"**Answer:** {response['answer']}")

            with st.expander("🔍 Dive into supporting documents"):
                for i, d in enumerate(response["context"]):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(d.page_content)
                    st.markdown("---")

except Exception as e:
    st.error(f"⚠️ Oops! Something went wrong: `{e}`. Please contact my Master 👨‍💻")

# Footer
st.markdown('<div class="footer">Made with ❤️ by Ananthmanoj using Langchain, HuggingFace & Streamlit</div>', unsafe_allow_html=True)
