import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# --- Page Config ---
st.set_page_config(page_title="RAG Data Assistant", page_icon="📚", layout="wide")
load_dotenv()

# --- Initialize Session States ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- Helper Functions ---
def process_documents(docs):
    """Splits documents, creates embeddings, and builds FAISS index"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create vector store and save it to session state (memory)
    vector_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)
    st.session_state.vector_store = vector_store
    return len(chunks)

# --- Sidebar: Data Ingestion UI ---
with st.sidebar:
    st.title("📂 Add Your Data")
    st.info("Upload a PDF or enter a Website URL to build your knowledge base.")
    
    # 1. PDF Upload
    uploaded_file = st.file_uploader("Upload a PDF Book/Document", type=["pdf"])
    if st.button("Process PDF") and uploaded_file is not None:
        with st.spinner("Extracting and processing PDF..."):
            # Streamlit files are in memory. We need to save it to a temp file for PyPDFLoader
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
                num_chunks = process_documents(docs)
                st.success(f"PDF Processed! Created {num_chunks} chunks.")
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
            finally:
                os.remove(tmp_file_path) # Clean up temp file

    st.divider()

    # 2. Website URL Input
    website_url = st.text_input("Or enter a Website URL:")
    if st.button("Process Website") and website_url:
        with st.spinner("Scraping and processing website..."):
            try:
                loader = WebBaseLoader(website_url)
                docs = loader.load()
                num_chunks = process_documents(docs)
                st.success(f"Website Processed! Created {num_chunks} chunks.")
            except Exception as e:
                st.error(f"Error reading website: {e}")

    st.divider()
    if st.button("🗑️ Clear Chat & Data"):
        st.session_state.messages = []
        st.session_state.vector_store = None
        st.rerun()

# --- Main Chat UI ---
st.title("💬 RAG Knowledge Assistant")

if st.session_state.vector_store is None:
    st.warning("👈 Please upload a PDF or provide a Website URL in the sidebar to start.")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if query := st.chat_input("Ask a question about your uploaded data..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Setup Retriever
                    retriever = st.session_state.vector_store.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
                    )
                    
                    # Setup LLM & Prompt
                    llm = ChatMistralAI(model="mistral-small")
                    prompt_tmpl = ChatPromptTemplate.from_messages([
                        ("system", "You are a helpful assistant. Use only the provided context to answer the user's question. If the answer is not present in the context, say you don't know."),
                        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
                    ])
                    
                    # 1. Retrieve Docs
                    docs = retriever.invoke(query)
                    context_text = "\n".join([doc.page_content for doc in docs])
                    
                    # 2. Build Prompt & Get Response
                    final_prompt = prompt_tmpl.invoke({"context": context_text, "question": query})
                    response = llm.invoke(final_prompt)
                    full_response = response.content
                    
                    # 3. Display Output
                    st.markdown(full_response)
                    with st.expander("🔍 View Retrieved Sources"):
                        st.write(context_text)
                    
                    # Save to history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")