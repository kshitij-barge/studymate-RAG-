import os
from langchain_community.document_loaders import WebBaseLoader

os.environ["USER_AGENT"] = "Mozilla/5.0"

url = "https://www.apple.com/in/macbook-pro/"

loader = WebBaseLoader(url)

docs = loader.load()

print(len(docs))
print(docs[0].page_content[:1000])