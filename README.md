# Ask about Nicolas — RAG Chatbot

A retrieval-augmented generation (RAG) chatbot that answers questions about Nicolas's
background, work experience and projects, grounded in the documents in [`data/`](data/).

- **LLM**: [Groq](https://groq.com) (`openai/gpt-oss-20b`, free tier)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (runs locally, no API key)
- **Vector store**: [Chroma](https://www.trychroma.com/) (in-memory, rebuilt on startup)
- **UI**: [Gradio](https://www.gradio.app/)

New to RAG? [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md) (in Spanish) walks through
`app.py` step by step — what each part does and why.

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

### What's in the knowledge base

| File | Contents |
|---|---|
| [`about-nicolas.md`](data/about-nicolas.md) | Bio, work experience, web projects, contact |
| [`ai-projects-overview.md`](data/ai-projects-overview.md) | The six AI projects at a glance, all live-demo URLs, the hardest deploys |
| `project-1…6-*.md` | One fact sheet per AI project: approach, numbers, bugs and fixes |
| [`lessons-and-skills.md`](data/lessons-and-skills.md) | The 8 cross-cutting lessons and skills |
| [`education-and-training.md`](data/education-and-training.md), [`work-experience.md`](data/work-experience.md), [`skills-and-languages.md`](data/skills-and-languages.md), [`personality-interests-and-goals.md`](data/personality-interests-and-goals.md) | Personal side: studies, jobs, skills with years, languages, interests, goals |
| [`faq-en.md`](data/faq-en.md), [`faq-es.md`](data/faq-es.md) | Question-and-answer pairs worded the way people actually ask |
| [`resumen-proyectos-ia-es.md`](data/resumen-proyectos-ia-es.md) | Spanish summary of the AI work, so Spanish questions retrieve well |

Writing tips that came out of testing retrieval (~80 chunks, k=4):

- **Start every paragraph with the topic name** (e.g. `Waste classifier, data and results: …`).
  Chunks are 800 characters and retrieved in isolation, so each one has to say what it's
  about on its own.
- **Keep paragraphs under ~750 characters** so the splitter doesn't cut them mid-fact.
- **Write the facts in English, and add a short Spanish summary.** The embedding model
  (`all-MiniLM-L6-v2`) is English-centric: Spanish questions only retrieve well against
  Spanish text.
- **Add question-and-answer paragraphs** (`Question: … Answer: …`) for things people ask
  directly. They match a question's wording far better than prose does: with them, a
  24-question benchmark went from 15/18-style misses to 21/24 hits within the top 4 chunks
  (23/24 with k=6), and Spanish questions in particular stopped missing.
- **Add overview paragraphs for cross-cutting questions** ("which project had the hardest
  deploy?", "what are all the demo links?"). Questions that span projects don't match any
  single project's sheet.

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
