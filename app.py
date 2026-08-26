import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
GROQ_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about Nicolas Alejandro Angel "
    "Robles based only on the context provided below. If the answer isn't in the "
    "context, say you don't have that information instead of guessing. Keep answers "
    "concise and friendly.\n\nContext:\n{context}"
)


def load_documents():
    docs = []
    for path in DATA_DIR.glob("**/*"):
        if path.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(path)).load())
        elif path.suffix.lower() in {".md", ".txt"}:
            docs.extend(TextLoader(str(path), encoding="utf-8").load())
    return docs


def build_vectorstore():
    documents = load_documents()
    if not documents:
        raise RuntimeError(
            f"No documents found in {DATA_DIR}. Add .md, .txt or .pdf files there."
        )
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma.from_documents(chunks, embeddings)


def build_chain():
    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.2)
    return retriever, llm


def make_answer_fn(retriever, llm):
    def answer(message, history):
        relevant_docs = retriever.invoke(message)
        context = "\n\n".join(doc.page_content for doc in relevant_docs)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
            *history,
            {"role": "user", "content": message},
        ]
        response = llm.invoke(messages)
        return response.content

    return answer


def main():
    retriever, llm = build_chain()
    demo = gr.ChatInterface(
        fn=make_answer_fn(retriever, llm),
        type="messages",
        title="Ask about Nicolas",
        description=(
            "A RAG chatbot that answers questions about Nicolas's background, "
            "experience and projects, grounded in his own bio documents."
        ),
        examples=[
            "What is Nicolas's professional background?",
            "What projects has Nicolas worked on?",
            "How can I contact Nicolas?",
        ],
    )
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    main()
