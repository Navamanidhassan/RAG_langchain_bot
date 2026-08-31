import streamlit as st

from main import ask_question

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(page_title="Book RAG Assistant", page_icon="📚", layout="wide")


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("📚 Book RAG Assistant")

st.write("Ask questions based on the content of the uploaded book.")


# ---------------------------------------------------------
# Sidebar - Upload Rules
# ---------------------------------------------------------

with st.sidebar:
    st.header("📌 Book Upload Rules")

    st.markdown(
        """
        **Please follow these rules when uploading a book:**

        1. Upload the book in **PDF format only**.
        
        2. The PDF should contain **readable/selectable text**.
        
        3. **Scanned image-only PDFs** may not work correctly.
        
        4. Do not upload **password-protected or encrypted PDFs**.
        
        5. Keep the PDF size reasonably small for faster processing.
        
        6. Upload **one book/document at a time**.
        """
    )

    st.divider()

    st.info(
        "The system retrieves relevant sections from the document "
        "and uses Mistral AI to generate the answer."
    )


# ---------------------------------------------------------
# Chat History
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------
# Display Chat History
# ---------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------
# User Question
# ---------------------------------------------------------

query = st.chat_input("Ask a question about the book...")


if query:
    # Display user question
    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.messages.append({"role": "user", "content": query})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching the document..."):
            result = ask_question(query)

        st.markdown(result["answer"])

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
