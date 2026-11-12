# ============================================================
# UNFILTERED.AI - HYBRID PDF RAG + GENERAL LLM CHATBOT
# Streamlit + FAISS + Sentence Transformers + Groq
# ============================================================

import os
import tempfile
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq


# ============================================================
# 1. APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Unfiltered.AI | PDF Assistant",
    page_icon="📚",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")


def get_config(name: str, default: str = "") -> str:
    """Read Streamlit Cloud secrets first, then local environment variables."""
    try:
        secret_value = st.secrets.get(name, default)
    except Exception:
        secret_value = default

    return str(os.getenv(name, secret_value)).strip()


GROQ_API_KEY = get_config("GROQ_API_KEY")
GROQ_MODEL = get_config("GROQ_MODEL", "openai/gpt-oss-20b")


# ============================================================
# 2. LOAD MODELS
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_llm(api_key: str, model_name: str):
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0.1,
        max_tokens=1024,
    )


# ============================================================
# 3. PDF INGESTION, CHUNKING AND VECTOR DATABASE
# ============================================================

def process_pdfs(uploaded_files):
    all_documents = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            safe_name = Path(uploaded_file.name).name
            file_path = Path(temp_dir) / safe_name
            file_path.write_bytes(uploaded_file.getvalue())

            loader = PyMuPDFLoader(str(file_path))
            documents = loader.load()

            for document in documents:
                document.metadata["source"] = safe_name

            all_documents.extend(documents)

    return all_documents


def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    return splitter.split_documents(documents)


def create_vector_database(chunks, embedding_model):
    texts = [chunk.page_content for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


# ============================================================
# 4. RESPONSE EXTRACTION
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
            elif isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))

        return "\n".join(text_parts).strip()

    return str(content).strip()


# ============================================================
# 5. HYBRID QUESTION ANSWERING: RAG OR GENERAL LLM
# ============================================================

def answer_question(
    query,
    index,
    chunks,
    embedding_model,
    llm,
    top_k=5,
):
    """
    If relevant PDF chunks are found, answer using RAG.
    Otherwise, send the question directly to the general-purpose LLM.

    Note: General LLM mode does not have live internet access.
    """

    if index is None or index.ntotal == 0:
        # No document index available: use general LLM mode.
        response = llm.invoke(
            "You are Unfiltered.AI, a helpful general-purpose AI assistant. "
            "Answer clearly and accurately. Do not claim to know live or "
            "real-time information unless it is provided in the conversation.\n\n"
            f"User question: {query}"
        )
        answer = extract_response_text(response)

        return {
            "answer": answer or "The AI model returned an empty response.",
            "sources": [],
            "mode": "general",
        }

    # Embed the user's question.
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
    ).astype(np.float32)

    faiss.normalize_L2(query_embedding)

    # Retrieve the closest PDF chunks.
    k = min(top_k, index.ntotal)
    distances, indices = index.search(query_embedding, k)

    results = []
    relevance_threshold = 0.35

    for score, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue

        if float(score) < relevance_threshold:
            continue

        results.append({
            "content": chunks[idx].page_content,
            "metadata": chunks[idx].metadata,
            "score": float(score),
        })

    # No relevant PDF context: answer with the LLM directly.
    if not results:
        general_prompt = f"""
You are Unfiltered.AI, a helpful general-purpose AI assistant.

Answer the user's question clearly and accurately.

Instructions:
- Use your general knowledge to answer.
- Explain concepts in simple language.
- Use examples when helpful.
- Do not pretend to have access to live information.
- For current weather, live prices, breaking news, or other real-time
  questions, explain that a live information source is required.
- If uncertain, say so honestly.

USER QUESTION:
{query}

ANSWER:
"""
        response = llm.invoke(general_prompt)
        answer = extract_response_text(response)

        return {
            "answer": answer or "The AI model returned an empty response.",
            "sources": [],
            "mode": "general",
        }

    # Relevant PDF context found: use RAG.
    context_parts = []
    sources = []

    for result in results:
        metadata = result["metadata"]
        source_name = metadata.get("source", "Unknown PDF")
        page = metadata.get("page", "Unknown")

        # PyMuPDFLoader page metadata is zero-based.
        if isinstance(page, int):
            page += 1

        context_parts.append(
            f"Source: {source_name}\n"
            f"Page: {page}\n"
            f"Content:\n{result['content']}"
        )

        sources.append({
            "source": source_name,
            "page": page,
            "score": result["score"],
        })

    context = "\n\n".join(context_parts)

    rag_prompt = f"""
You are Unfiltered.AI, an assistant answering questions from uploaded PDFs.

Answer the user's question using ONLY the information in the context below.

Rules:
1. Do not invent facts.
2. Do not use outside knowledge to fill gaps.
3. If the answer is not available in the context, say so clearly.
4. Explain in simple, understandable language.
5. Use bullet points when helpful.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    response = llm.invoke(rag_prompt)
    answer = extract_response_text(response)

    return {
        "answer": answer or "The AI model returned an empty response.",
        "sources": sources,
        "mode": "rag",
    }


# ============================================================
# 6. SESSION STATE
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
# 7. SIDEBAR: PDF UPLOAD AND PROCESSING
# ============================================================

with st.sidebar:
    st.title("📚 Unfiltered.AI")
    st.caption("Hybrid PDF RAG + General AI Assistant")
    st.divider()

    st.subheader("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload your PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    process_button = st.button(
        "Process PDFs",
        use_container_width=True,
        type="primary",
    )

    if process_button:
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")

        elif not GROQ_API_KEY:
            st.error(
                "GROQ_API_KEY is missing. Add it in Streamlit Cloud "
                "Settings → Secrets, or in your local .env file."
            )

        else:
            try:
                with st.spinner("Loading embedding model..."):
                    embedding_model = load_embedding_model()

                with st.spinner("Loading Groq LLM..."):
                    load_llm(GROQ_API_KEY, GROQ_MODEL)

                with st.spinner("Reading PDF documents..."):
                    documents = process_pdfs(uploaded_files)

                if not documents:
                    st.error("No readable content found in the PDFs.")
                    st.stop()

                with st.spinner("Creating text chunks..."):
                    chunks = create_chunks(documents)

                if not chunks:
                    st.error("No text chunks were generated from the PDFs.")
                    st.stop()

                with st.spinner("Creating FAISS vector database..."):
                    index = create_vector_database(chunks, embedding_model)

                st.session_state.faiss_index = index
                st.session_state.text_chunks = chunks
                st.session_state.uploaded_pdf_names = [
                    file.name for file in uploaded_files
                ]
                st.session_state.chat_history = []

                st.success("PDFs processed successfully!")
                st.rerun()

            except Exception as error:
                st.error(f"PDF processing failed: {error}")

    st.divider()

    if st.session_state.faiss_index is not None:
        st.success("Vector database ready")

        st.write(
            f"**PDF files:** {len(st.session_state.uploaded_pdf_names)}"
        )
        st.write(
            f"**Text chunks:** {len(st.session_state.text_chunks)}"
        )

        if st.button("Clear documents", use_container_width=True):
            st.session_state.faiss_index = None
            st.session_state.text_chunks = None
            st.session_state.chat_history = []
            st.session_state.uploaded_pdf_names = []
            st.rerun()


# ============================================================
# 8. MAIN CHAT INTERFACE
# ============================================================

st.title("🤖 Chat with your PDFs and General AI")

st.write(
    "Ask questions about your uploaded PDFs or ask general questions. "
    "Unfiltered.AI uses document retrieval when relevant PDF context is found "
    "and falls back to Groq for general questions."
)

if not GROQ_API_KEY:
    st.warning(
        "Groq API key is not configured. Add GROQ_API_KEY in "
        "Streamlit Cloud Settings → Secrets."
    )

if st.session_state.faiss_index is None:
    st.info(
        "Upload and process PDFs to enable document-based questions. "
        "After processing, you can also ask general AI questions."
    )

    st.markdown("### How it works")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📄 Upload")
        st.write("Upload one or more PDF documents.")

    with col2:
        st.markdown("### 🔍 Retrieve")
        st.write("FAISS finds relevant information in your PDFs.")

    with col3:
        st.markdown("### 💬 Ask")
        st.write("Groq answers using PDF context or general knowledge.")

else:
    embedding_model = load_embedding_model()
    llm = load_llm(GROQ_API_KEY, GROQ_MODEL) if GROQ_API_KEY else None

    with st.expander("View uploaded documents"):
        for name in st.session_state.uploaded_pdf_names:
            st.write(f"📄 {name}")

    st.divider()

    # Restore previous conversation.
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                mode = message.get("mode")
                if mode == "rag":
                    st.caption("📚 Answered using your uploaded PDFs (RAG)")
                elif mode == "general":
                    st.caption("🤖 Answered using General AI (Groq LLM)")

                for source in message.get("sources", []):
                    with st.expander(
                        f"Source: {source['source']} | Page: {source['page']}"
                    ):
                        st.write(f"Similarity: {source['score']:.4f}")

    query = st.chat_input("Ask about your PDFs or ask a general question...")

    if query:
        if not GROQ_API_KEY or llm is None:
            st.error(
                "Groq API key is missing. Configure GROQ_API_KEY in "
                "Streamlit Cloud Settings → Secrets."
            )
        else:
            st.session_state.chat_history.append({
                "role": "user",
                "content": query,
            })

            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = answer_question(
                            query=query,
                            index=st.session_state.faiss_index,
                            chunks=st.session_state.text_chunks,
                            embedding_model=embedding_model,
                            llm=llm,
                        )

                        if result["mode"] == "rag":
                            st.caption(
                                "📚 Answered using your uploaded PDFs (RAG)"
                            )
                        else:
                            st.caption(
                                "🤖 Answered using General AI (Groq LLM)"
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
                            "sources": result["sources"],
                            "mode": result["mode"],
                        })

                    except Exception as error:
                        error_message = f"Answer generation failed: {error}"
                        st.error(error_message)

                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_message,
                            "sources": [],
                            "mode": "error",
                        })
