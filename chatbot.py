
import streamlit as st
from groq import Groq
from pypdf import PdfReader
import re

st.set_page_config(page_title="RAG AI Chatbot", page_icon="🤖")

st.title("🤖 RAG-Based AI Chatbot")
st.caption("Ask questions from your PDF documents using Groq AI")

# API KEY
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter Groq API Key", type="password")
    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process Documents"):
        if not uploaded_files:
            st.warning("Please upload a PDF first.")
        else:
            chunks = []

            for file in uploaded_files:
                reader = PdfReader(file)

                for page_no, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    words = text.split()

                    for i in range(0, len(words), 180):
                        chunk = " ".join(words[i:i+180])

                        if chunk.strip():
                            chunks.append({
                                "text": chunk,
                                "source": file.name,
                                "page": page_no + 1
                            })

            st.session_state.documents = chunks
            st.success(f"Processed {len(chunks)} text chunks!")

# CHAT HISTORY
if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []

# RETRIEVAL
def retrieve_documents(question, documents, top_k=3):
    query_words = set(re.findall(r"\w+", question.lower()))

    scored = []

    for doc in documents:
        doc_words = set(
            re.findall(r"\w+", doc["text"].lower())
        )
        score = len(query_words & doc_words)

        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [doc for score, doc in scored[:top_k] if score > 0]

# DISPLAY CHAT
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# USER QUESTION
question = st.chat_input("Ask a question about your PDF...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    if not api_key:
        st.error("Please enter your Groq API key in Settings.")

    elif not st.session_state.documents:
        st.warning("Please upload and process a PDF first.")

    else:
        with st.chat_message("assistant"):
            try:
                relevant_docs = retrieve_documents(
                    question,
                    st.session_state.documents
                )

                if not relevant_docs:
                    st.warning(
                        "No relevant information found in the PDF."
                    )
                else:
                    context = "\n\n".join(
                        f"Source: {doc['source']}, "
                        f"Page: {doc['page']}\n{doc['text']}"
                        for doc in relevant_docs
                    )

                    client = Groq(api_key=api_key)

                    response = client.chat.completions.create(
                        model="openai/gpt-oss-20b",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Answer using the supplied document "
                                    "context. If the answer is not in the "
                                    "context, say you could not find it."
                                )
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"Document context:\n{context}\n\n"
                                    f"Question: {question}"
                                )
                            }
                        ],
                        temperature=0.3,
                        max_tokens=1024
                    )

                    answer = response.choices[0].message.content

                    st.markdown(answer)

                    st.caption(
                        "Sources: " + ", ".join(
                            f"{d['source']} (page {d['page']})"
                            for d in relevant_docs
                        )
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

            except Exception as e:
                st.error(f"Error: {e}")

with st.sidebar:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
