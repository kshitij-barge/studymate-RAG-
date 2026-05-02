from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

print("Loading PDF...")
loader = PyPDFLoader("documentloaders/deeplearning.pdf")
docs = loader.load()

print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)

print(f"Total chunks created: {len(chunks)}")

print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Creating FAISS vector store...")
vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embedding_model
)

print("Saving vector store locally...")
vector_store.save_local("vectorstore")

print("Vector database created successfully!")