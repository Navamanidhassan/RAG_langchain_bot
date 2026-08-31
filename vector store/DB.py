from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

from langchain_core import documents

docs=[
    documents.Document(page_content="i am a data scientist",metadata={"category":"science"}),
    documents.Document(page_content="i am a data analyst",metadata={"category":"analysis"}),
    documents.Document(page_content="i am a data engineer",metadata={"category":"engineer"})
]

embedding_model=MistralAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory='chroma_db'
    )

query="who is the data scientist?"

result=vectorstore.similarity_search(query,k=1)

for r in result:
    print(r.page_content)
    print(r.metadata)

retiver=vectorstore.as_retriever()

docs=retiver.invoke('what is analysis?')

for d in docs:
    print(d.page_content)