import os
import tempfile
import streamlit as st

from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.vector_store import create_vector_store
from src.rag import generate_answer


st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Research Paper Assistant")

st.markdown("""
Upload one or more research papers and ask questions.

The system will:
- Search across all uploaded PDFs
- Generate answers using Gemini
- Show source PDF, page number and paragraph
""")

# ----------------------------------------------------
# Session State Initialization
# ----------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------------------------------
# Upload PDFs
# ----------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type=["pdf"],
    accept_multiple_files=True
)

# ----------------------------------------------------
# Process PDFs Only Once
# ----------------------------------------------------

if uploaded_files:

    if "db" not in st.session_state:

        with st.spinner("Processing documents..."):

            all_documents = []

            for uploaded_file in uploaded_files:

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    pdf_path = tmp_file.name

                try:
                    documents = load_pdf(pdf_path)
                finally:
                    os.remove(pdf_path)

                for doc in documents:
                    doc.metadata["pdf_name"] = uploaded_file.name

                all_documents.extend(documents)

            chunks = split_documents(all_documents)

            for chunk in chunks:
                if "pdf_name" not in chunk.metadata:
                    chunk.metadata["pdf_name"] = "Unknown"

            db = create_vector_store(chunks)

            st.session_state.db = db
            st.session_state.num_docs = len(uploaded_files)
            st.session_state.num_chunks = len(chunks)

        st.success("Documents processed successfully!")

    st.info(
        f"Loaded {st.session_state.num_docs} PDFs | "
        f"{st.session_state.num_chunks} chunks indexed"
    )

    # ------------------------------------------------
    # Display Previous Chat History
    # ------------------------------------------------

    for chat in st.session_state.chat_history:

        with st.chat_message("user"):
            st.write(chat["question"])

        with st.chat_message("assistant"):
            st.write(chat["answer"])

            with st.expander("View Sources"):

                for i, doc in enumerate(
                    chat["sources"],
                    start=1
                ):

                    page_no = (
                        doc.metadata.get("page", 0)
                        + 1
                    )

                    pdf_name = doc.metadata.get(
                        "pdf_name",
                        "Unknown PDF"
                    )

                    st.markdown(
                        f"### Source {i}"
                    )

                    st.write(
                        f"📄 **PDF:** {pdf_name}"
                    )

                    st.write(
                        f"📄 **Page:** {page_no}"
                    )

                    st.markdown(
                        "**Retrieved Paragraph:**"
                    )

                    st.info(
                        doc.page_content
                    )

    # ------------------------------------------------
    # Chat Input
    # ------------------------------------------------

    question = st.chat_input(
        "Ask a question about the uploaded documents..."
    )

    if question:

        # Show User Message Immediately

        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Generating answer..."):

            answer, docs = generate_answer(
                question,
                st.session_state.db
            )

        # Save Conversation

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": answer,
                "sources": docs
            }
        )

        # Show Assistant Response

        with st.chat_message("assistant"):
            st.write(answer)

            with st.expander("View Sources"):

                for i, doc in enumerate(
                    docs,
                    start=1
                ):

                    page_no = (
                        doc.metadata.get("page", 0)
                        + 1
                    )

                    pdf_name = doc.metadata.get(
                        "pdf_name",
                        "Unknown PDF"
                    )

                    st.markdown(
                        f"### Source {i}"
                    )

                    st.write(
                        f"📄 **PDF:** {pdf_name}"
                    )

                    st.write(
                        f"📄 **Page:** {page_no}"
                    )

                    st.markdown(
                        "**Retrieved Paragraph:**"
                    )

                    st.info(
                        doc.page_content
                    )

# ----------------------------------------------------
# Reset Button
# ----------------------------------------------------

st.divider()

if st.button("Clear Documents"):

    keys_to_remove = [
        "db",
        "num_docs",
        "num_chunks",
        "chat_history"
    ]

    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]

    st.rerun()