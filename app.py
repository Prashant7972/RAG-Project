import time

import faiss
import numpy as np
import streamlit as st
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

MAX_MB = 5
MAX_QUESTIONS = 15

# Fast model first, stronger model as backup.
# If you get a "404 NOT_FOUND" error, the message tells you the replacement name.
MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]

st.set_page_config(page_title="Chat with your document", page_icon="📄")
st.title("📄 Chat with your document")
st.caption("Don't upload sensitive documents. Files are processed in memory only.")


@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def get_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


def read_file(f):
    if f.name.lower().endswith(".pdf"):
        reader = PdfReader(f)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return f.read().decode("utf-8", errors="ignore")


def chunk_text(text, size=800, overlap=150):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c for c in chunks if c.strip()]


def build_index(chunks, embedder):
    emb = embedder.encode(chunks, normalize_embeddings=True)
    emb = np.array(emb, dtype="float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


def stream_answer(question, history, index, chunks, embedder):
    """Yields the answer piece by piece so it appears while it's being written."""
    # 1. Find relevant chunks (include the previous question so follow-ups work)
    previous_user_msgs = [m["content"] for m in history if m["role"] == "user"]
    search_text = (previous_user_msgs[-1] + " " + question) if previous_user_msgs else question
    q = embedder.encode([search_text], normalize_embeddings=True).astype("float32")
    k = min(4, len(chunks))
    _, ids = index.search(q, k)
    context = "\n\n".join(chunks[i] for i in ids[0])

    # 2. Short conversation memory (smaller prompt = faster answer)
    recent = history[-4:]
    convo = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in recent
    )

    prompt = (
        "Answer the user's question using only the document context below. "
        "Be concise. If the answer isn't in the context, say you couldn't find it "
        "in the document.\n\n"
        f"Document context:\n{context}\n\n"
        f"Conversation so far:\n{convo}\n\n"
        f"Question: {question}"
    )

    # 3. Ask Gemini with streaming, with a quick retry and a backup model
    client = get_client()
    last_error = None

    for model in MODELS:
        for attempt in range(2):
            started = False
            try:
                for chunk in client.models.generate_content_stream(
                    model=model, contents=prompt
                ):
                    if chunk.text:
                        started = True
                        yield chunk.text
                return
            except Exception as e:
                if started:
                    raise  # failed halfway through an answer
                last_error = e
                msg = str(e)
                if "404" in msg:
                    break  # model not available, go to the backup model
                if "503" in msg or "429" in msg:
                    time.sleep(1)
                    continue
                raise  # other errors (like a bad API key) shouldn't be retried

    raise last_error


embedder = load_embedder()
st.session_state.setdefault("messages", [])
st.session_state.setdefault("q_count", 0)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Your document")
    uploaded = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.q_count = 0
        st.rerun()
    if uploaded:
        left = MAX_QUESTIONS - st.session_state.q_count
        st.caption(f"Questions left this session: {max(left, 0)}")

# ---------- Process the uploaded file ----------
if uploaded:
    if uploaded.size > MAX_MB * 1024 * 1024:
        st.error(f"File too large. Max {MAX_MB} MB.")
        st.stop()

    if st.session_state.get("file_name") != uploaded.name:
        with st.spinner("Reading and indexing your document..."):
            text = read_file(uploaded)
            chunks = chunk_text(text)
            if not chunks:
                st.error("Couldn't extract any text from this file.")
                st.stop()
            st.session_state.chunks = chunks
            st.session_state.index = build_index(chunks, embedder)
            st.session_state.file_name = uploaded.name
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": f"I've read **{uploaded.name}**. Ask me anything about it!",
                }
            ]
            st.session_state.q_count = 0
else:
    st.info("👈 Upload a PDF or TXT file in the sidebar to start chatting.")

# ---------- Show the chat history ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------- Chat input ----------
user_input = st.chat_input(
    "Ask a question about your document...",
    disabled=uploaded is None,
)

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    if st.session_state.q_count >= MAX_QUESTIONS:
        with st.chat_message("assistant"):
            st.warning("Question limit reached for this session. Refresh the page to start again.")
    else:
        history = list(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("assistant"):
            try:
                reply = st.write_stream(
                    stream_answer(
                        user_input,
                        history,
                        st.session_state.index,
                        st.session_state.chunks,
                        embedder,
                    )
                )
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.session_state.q_count += 1
            except Exception as e:
                print("Gemini error:", e)  # shows in your terminal, not to visitors
                st.error("The AI service is busy or unavailable. Please try again in a minute.")