from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
data = PyPDFLoader("documentloaders/GRU.pdf")

docs = data.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)

chunks = splitter.split_documents(docs) 

print(len(chunks))