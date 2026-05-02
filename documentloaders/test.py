from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

spitter = CharacterTextSplitter(separator="" ,chunk_size=10, chunk_overlap=1)
data = TextLoader("documentloaders/notes.txt")
docs = data.load()

chunks = spitter.split_documents(docs)

print(len(chunks))

for i in chunks:
    print(i.page_content)
    print("----")
    
# print(docs[0].page_content)