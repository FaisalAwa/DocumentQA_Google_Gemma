# import os
# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS  # vector store db
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings  # vector embedding technique
# from dotenv import load_dotenv

# load_dotenv()

# # Load the groq and google generative ai embeddings
# groq_api_key = os.getenv("GROQ_API_KEY")
# os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

# st.title("Gemma Model Document Q & A")

# llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-it")

# prompt = ChatPromptTemplate.from_template(
#     """
#     Answer the questions based on the provided context only.
#     Please provide the most accurate response based on the question
#     <context>
#     {context}
#     <context>
#     Questions: {input}
#     """
# )

# def vec_embedding():
#     if "vectors" not in st.session_state:
#         st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#         st.session_state.loader = PyPDFDirectoryLoader("./books")  # Data Ingestion
#         st.session_state.docs = st.session_state.loader.load()  # Document Loading
#         st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
#         st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# prompt1 = st.text_input("What do you want to ask from the documents?")

# if st.button("Creating Vector Store"):
#     vec_embedding()
#     st.write("Vector Store DB is Ready")

# import time

# if prompt1:
#     if "vectors" not in st.session_state:
#         vec_embedding()
#     documents_chain = create_stuff_documents_chain(llm, prompt)
#     retriever = st.session_state.vectors.as_retriever()
#     retrieval_chain = create_retrieval_chain(retriever, documents_chain)  # Corrected typo

#     start = time.process_time()
#     response = retrieval_chain.invoke({'input': prompt1})
#     st.write(response['answer'])

#     # With a streamlit expander
#     with st.expander("Document Similarity Search"):
#         # Find the relevant chunks
#         for i, doc in enumerate(response["context"]):
#             st.write(doc.page_content)
#             st.write("--------------------------")


import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS  # vector store db
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # vector embedding technique
from dotenv import load_dotenv
import time

load_dotenv()

# Load the groq and google generative ai embeddings
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="Gemma Model Document Q & A", layout="wide")

# Custom CSS for styling
def add_custom_css():
    st.markdown(
        """
        <style>
        .header {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #FFFF00;
        }
        .subheader {
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            color: #FF0000;
        }
        .description {
            font-size: 18px;
            color: #AAAAAA;
        }
        body {
            background-color: #001f3f;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_custom_css()

# SVG Animation
st.markdown("""
<svg width="100%" height="200" xmlns="http://www.w3.org/2000/svg">
  <g>
    <title>Document Q & A</title>
    <rect width="100%" height="200" fill="#001f3f" />
    <circle cx="150" cy="100" r="80" fill="#FF0000" />
    <text x="150" y="115" font-size="35" font-family="Arial" fill="#FFFFFF" text-anchor="middle">📚</text>
    <text x="50%" y="180" font-size="24" font-family="Arial" fill="#FFFF00" text-anchor="middle">
      <animate attributeName="opacity" values="0;1;0" dur="3s" repeatCount="indefinite" />
      Gemma Model Document Q & A
    </text>
  </g>
</svg>
""", unsafe_allow_html=True)

st.markdown('<div class="header">Gemma Model Document Q & A 📚🤖</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Retrieve and Analyze Information from Your Documents</div>', unsafe_allow_html=True)

st.write("""
### Project Description:
This application uses the power of the Gemma model to retrieve and analyze information from your documents. Simply ask a question, and the model will search the most relevant information from the documents in the backend. Additionally, it performs a similarity search across all documents to provide the best answers.
""")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-it")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

def vec_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./books")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

prompt1 = st.text_input("What do you want to ask from the documents?")

if st.button("Creating Vector Store"):
    vec_embedding()
    st.write("Vector Store DB is Ready")

if prompt1:
    if "vectors" not in st.session_state:
        vec_embedding()
    documents_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, documents_chain)  # Corrected typo

    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.subheader("The Response is:")
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------")
