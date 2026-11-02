# 📄 Chat with your document (RAG)

🔗 **Live demo:** https://your-app.streamlit.app

Upload a PDF or TXT file and ask questions about it.
The app may take ~30 seconds to wake up if it hasn't been used recently.

## How it works
1. The document is split into chunks
2. Chunks are embedded with fastembed and stored in a FAISS index
3. Your question retrieves the most relevant chunks
4. Gemini answers using only those chunks, streamed live

## Tech stack
Streamlit, FAISS, fastembed, Google Gemini API

## Run locally
pip install -r requirements.txt
Add GEMINI_API_KEY to .streamlit/secrets.toml
streamlit run app.py