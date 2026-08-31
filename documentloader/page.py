from langchain_community.document_loaders import WebBaseLoader

url = "https://vit.ac.in/"
data = WebBaseLoader(url)

docs = data.load()

print(len(docs))

print(docs[0])
