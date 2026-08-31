from langchain_community.document_loaders import TextLoader

data = TextLoader("documentloader/notes.txt")

docs=data.load()

print(len(docs))