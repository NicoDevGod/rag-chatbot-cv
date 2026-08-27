# Ask about Nicolas — RAG Chatbot

A retrieval-augmented generation (RAG) chatbot that answers questions about Nicolas's
background, work experience and projects, grounded in the documents in [`data/`](data/).

- **LLM**: [Groq](https://groq.com) (`openai/gpt-oss-20b`, free tier)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (runs locally, no API key)
- **Vector store**: [Chroma](https://www.trychroma.com/) (in-memory, rebuilt on startup)
- **UI**: [Gradio](https://www.gradio.app/)

New to RAG? [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md) walks through `app.py`
step by step — what each part does and why.

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

## Deploying to Render

This repo includes a [`render.yaml`](render.yaml) Blueprint, so Render can configure
everything automatically:

1. Sign in at https://dashboard.render.com (GitHub login works well).
2. **New → Blueprint**, pick this repo. Render reads `render.yaml` and proposes a
   free web service named `rag-chatbot-cv`.
3. When prompted for the `GROQ_API_KEY` environment variable, paste your key.
4. Deploy. The first build takes a few minutes (installing `torch`, `transformers`, etc.).

Render's free tier sleeps the service after 15 minutes of inactivity — the next visit
takes 30-60s to wake up, then responds normally.

Once it's live, the service URL (`https://rag-chatbot-cv-xxxx.onrender.com`) is the
link to use in the portfolio.
