# Ask about Nicolas — RAG Chatbot

A retrieval-augmented generation (RAG) chatbot that answers questions about Nicolas's
background, work experience and projects, grounded in the documents in [`data/`](data/).

- **LLM**: [Groq](https://groq.com) (`llama-3.1-8b-instant`, free tier)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (runs locally, no API key)
- **Vector store**: [Chroma](https://www.trychroma.com/) (in-memory, rebuilt on startup)
- **UI**: [Gradio](https://www.gradio.app/)

## Local setup

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

2. Get a free Groq API key at https://console.groq.com/keys and add it to a `.env` file:

   ```bash
   cp .env.example .env
   # then edit .env and paste your key
   ```

3. Run the app:

   ```bash
   python app.py
   ```

   Gradio will print a local URL (usually http://127.0.0.1:7860).

## Adding your own content

Drop `.md`, `.txt` or `.pdf` files into [`data/`](data/) — they're automatically loaded,
chunked and embedded on startup. No code changes needed.

## Deploying to Hugging Face Spaces

1. Create a new Space at https://huggingface.co/new-space — SDK: **Gradio**, hardware: **CPU basic (free)**.
2. Push this repo's contents to the Space's git remote (Spaces are git repos).
3. In the Space's **Settings → Repository secrets**, add `GROQ_API_KEY` with your key.
4. The Space builds automatically from `requirements.txt` and runs `app.py`.

Once it's live, the Space URL (`https://huggingface.co/spaces/<user>/<space-name>`) is
the link to use in the portfolio.
