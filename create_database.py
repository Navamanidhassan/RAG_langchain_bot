from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


# ---------------------------------------------------------
# Load PDF
# ---------------------------------------------------------

data = PyPDFLoader(
    "documentloader/Task Force Reports - Study on Semiconductor Design, Embedded Software and Services Industry.pdf"
)

docs = data.load()


# ---------------------------------------------------------
# Split Documents
# ---------------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)


# ---------------------------------------------------------
# Create Embedding Model
# ---------------------------------------------------------

embedding_model = MistralAIEmbeddings()


# ---------------------------------------------------------
# Create ChromaDB
# ---------------------------------------------------------

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)


print("Database created successfully.")
print(f"Total pages: {len(docs)}")
print(f"Total chunks: {len(chunks)}")