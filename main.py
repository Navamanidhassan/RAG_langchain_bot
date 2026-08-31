from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

load_dotenv()

embedding_model = MistralAIEmbeddings()

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model,
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

llm = ChatMistralAI()

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant that answers questions based on the context/document provided.
If the answer is not present let the user know that it is not present in the document.
answer in 3-4 sentences."""
    ),
    (
        "user",
        """
Document:{context}
Question:{question}
"""
    )
])


def ask_question(query):

    # Retrieve documents
    docs = retriever.invoke(query)

    context = "".join(
        [doc.page_content for doc in docs]
    )

    # Create final prompt
    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )

    # Generate response
    response = llm.invoke(final_prompt)

    return {
        "answer": response.content
    }