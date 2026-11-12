
# ============================================================
# PDF CHATBOT - RAG APPLICATION
# Streamlit + FAISS + Sentence Transformers + Groq
# ============================================================

import os
import tempfile
from pathlib import Path

import streamlit as st
import numpy as np
import faiss

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq


# ============================================================
# STEP 1: APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PDF Chatbot | RAG",
    page_icon="📚",
    layout="wide"
)


PROJECT_ROOT = Path(__file__).parent

load_dotenv(PROJECT_ROOT / ".env")

def get_config(name, default=""):
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default

    return os.getenv(name, value)

GROQ_API_KEY = get_config("GROQ_API_KEY").strip()

GROQ_MODEL = get_config(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
).strip()




# ============================================================
# STEP 2: LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# STEP 3: LOAD GROQ LLM
# ============================================================

@st.cache_resource
def load_llm(api_key, model_name):

    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0.1,
        max_tokens=1024
    )


# ============================================================
# STEP 4: PDF PROCESSING
# ============================================================

def process_pdfs(uploaded_files):

    all_documents = []

    with tempfile.TemporaryDirectory() as temp_dir:

        for uploaded_file in uploaded_files:

            file_path = Path(temp_dir) / uploaded_file.name

            with open(file_path, "wb") as file:
                file.write(uploaded_file.getvalue())

            loader = PyMuPDFLoader(str(file_path))

            documents = loader.load()

            for document in documents:

                # Keep the original uploaded filename
                document.metadata["source"] = uploaded_file.name

            all_documents.extend(documents)

    return all_documents


# ============================================================
# STEP 5: TEXT CHUNKING
# ============================================================

def create_chunks(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    return chunks


# ============================================================
# STEP 6: CREATE FAISS VECTOR DATABASE
# ============================================================

def create_vector_database(chunks, embedding_model):

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True
    ).astype(np.float32)

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ============================================================
# STEP 7: RESPONSE EXTRACTION
# ============================================================

def extract_response_text(response):

    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, str):
                text_parts.append(item)

            elif isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))

        return "\n".join(text_parts).strip()

    return str(content).strip()


# ============================================================
# STEP 8: RAG QUESTION ANSWERING
# ============================================================

def answer_question(
    query,
    index,
    chunks,
    embedding_model,
    llm,
    top_k=5
):

    # Create query embedding
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype(np.float32)

    faiss.normalize_L2(query_embedding)

    # Search FAISS
    k = min(top_k, index.ntotal)

    distances, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for score, idx in zip(distances[0], indices[0]):

        if idx == -1:
            continue

        # Basic relevance threshold
        if score < 0.2:
            continue

        results.append({
            "content": chunks[idx].page_content,
            "metadata": chunks[idx].metadata,
            "score": float(score)
        })

    if not results:

        return {
            "answer": (
                "I couldn't find relevant information "
                "in your uploaded PDFs."
            ),
            "sources": []
        }

    # Prepare context
    context_parts = []

    sources = []

    for result in results:

        metadata = result["metadata"]

        source_name = metadata.get(
            "source",
            "Unknown PDF"
        )

        page = metadata.get("page", "Unknown")

        if isinstance(page, int):
            page += 1

        context_parts.append(
            f"""
Source: {source_name}
Page: {page}

Content:
{result['content']}
"""
        )

        sources.append({
            "source": source_name,
            "page": page,
            "score": result["score"]
        })

    context = "\n\n".join(context_parts)

    # Prepare prompt
    prompt = f"""
You are a helpful PDF-based AI assistant.

Answer the user's question using ONLY the
information available in the provided context.

Rules:
1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not available in the PDFs,
   clearly say so.
4. Explain in simple and understandable language.
5. Use bullet points when helpful.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    # Generate answer
    response = llm.invoke(prompt)

    answer = extract_response_text(response)

    if not answer:
        answer = "The AI model returned an empty response."

    return {
        "answer": answer,
        "sources": sources
    }


# ============================================================
# STEP 9: INITIALIZE SESSION STATE
# ============================================================

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_pdf_names" not in st.session_state:
    st.session_state.uploaded_pdf_names = []


# ============================================================
# STEP 10: SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📚 PDF Chatbot")

    st.caption("Powered by Retrieval Augmented Generation")

    st.divider()

    st.subheader("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload your PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    process_button = st.button(
        "Process PDFs",
        use_container_width=True,
        type="primary"
    )

    if process_button:

        if not uploaded_files:

            st.warning("Please upload at least one PDF.")

        elif not GROQ_API_KEY:

            st.error("GROQ_API_KEY is missing from your .env file.")

        else:

            try:

                with st.spinner("Loading embedding model..."):

                    embedding_model = load_embedding_model()

                with st.spinner("Loading Groq LLM..."):

                    llm = load_llm(
                        GROQ_API_KEY,
                        GROQ_MODEL
                    )

                with st.spinner("Reading PDF documents..."):

                    documents = process_pdfs(uploaded_files)

                if not documents:
                    st.error("No readable content found in PDFs.")
                    st.stop()

                with st.spinner("Creating text chunks..."):

                    chunks = create_chunks(documents)

                if not chunks:
                    st.error("No text chunks were generated.")
                    st.stop()

                with st.spinner("Creating FAISS vector database..."):

                    index = create_vector_database(
                        chunks,
                        embedding_model
                    )

                # Save in session
                st.session_state.faiss_index = index

                st.session_state.text_chunks = chunks

                st.session_state.uploaded_pdf_names = [
                    file.name for file in uploaded_files
                ]

                # Reset previous chat
                st.session_state.chat_history = []

                st.success("PDFs processed successfully!")

                st.rerun()

            except Exception as e:

                st.error(f"Processing failed: {str(e)}")

    st.divider()

    if st.session_state.faiss_index is not None:

        st.success("Vector database ready")

        st.write(
            f"**PDF files:** "
            f"{len(st.session_state.uploaded_pdf_names)}"
        )

        st.write(
            f"**Text chunks:** "
            f"{len(st.session_state.text_chunks)}"
        )

        if st.button("Clear documents", use_container_width=True):

            st.session_state.faiss_index = None
            st.session_state.text_chunks = None
            st.session_state.chat_history = []
            st.session_state.uploaded_pdf_names = []

            st.rerun()


# ============================================================
# STEP 11: MAIN CHAT INTERFACE
# ============================================================

st.title("🤖 Chat with your PDFs")

st.write(
    "Upload your documents and ask questions. "
    "Your AI assistant will find relevant information "
    "and generate answers from your PDFs."
)

if st.session_state.faiss_index is None:

    st.info(
        "Upload one or more PDF files from the sidebar "
        "and click Process PDFs to get started."
    )

    st.markdown("### How it works")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📄")
        st.subheader("Upload")
        st.write("Upload multiple PDF documents.")

    with col2:
        st.markdown("### 🔍")
        st.subheader("Retrieve")
        st.write("FAISS searches for relevant information.")

    with col3:
        st.markdown("### 💬")
        st.subheader("Ask")
        st.write("Groq generates answers using retrieved context.")

else:

    # Load models
    embedding_model = load_embedding_model()

    llm = load_llm(
        GROQ_API_KEY,
        GROQ_MODEL
    )

    # Display uploaded documents
    with st.expander("View uploaded documents"):

        for name in st.session_state.uploaded_pdf_names:
            st.write(f"📄 {name}")

    st.divider()

    # Display chat history
    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message["role"] == "assistant":

                for source in message.get("sources", []):

                    with st.expander(
                        f"Source: {source['source']} | "
                        f"Page: {source['page']}"
                    ):

                        st.write(
                            f"Similarity: {source['score']:.4f}"
                        )

    # Chat input
    query = st.chat_input(
        "Ask something about your PDFs..."
    )

    if query:

        # Display user question
        st.session_state.chat_history.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.markdown(query)

        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner("Searching your documents..."):

                try:

                    result = answer_question(
                        query=query,
                        index=st.session_state.faiss_index,
                        chunks=st.session_state.text_chunks,
                        embedding_model=embedding_model,
                        llm=llm
                    )

                    st.markdown(result["answer"])

                    if result["sources"]:

                        st.markdown("**Sources:**")

                        for source in result["sources"]:

                            st.caption(
                                f"📄 {source['source']} | "
                                f"Page {source['page']} | "
                                f"Similarity: {source['score']:.4f}"
                            )

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"]
                    })

                except Exception as e:

                    st.error(f"Answer generation failed: {str(e)}")

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"An error occurred: {str(e)}",
                        "sources": []
                    })
