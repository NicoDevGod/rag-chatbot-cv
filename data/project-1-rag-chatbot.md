# Project 1: RAG chatbot (this chatbot)

RAG chatbot: This chatbot is Nicolas's first AI project. It answers questions about his experience using his own documents through retrieval-augmented generation (RAG). Code: https://github.com/NicoDevGod/rag-chatbot-cv. Live demo: https://rag-chatbot-cv-zre6.onrender.com.

RAG chatbot, how it works: A plain LLM only knows what it memorized in training, so it knows nothing about Nicolas. RAG fixes that: his documents are indexed as vectors, and for every question the 4 most semantically similar chunks are retrieved and inserted into the prompt. The model is never retrained; it only reads the retrieved context at question time.

RAG chatbot, indexing: At startup it loads the Markdown, text and PDF files from the data folder, splits them into chunks of 800 characters with 100 characters of overlap (LangChain RecursiveCharacterTextSplitter), creates embeddings with the model sentence-transformers/all-MiniLM-L6-v2 running through FastEmbed (ONNX), and stores them in an in-memory Chroma vector store.

RAG chatbot, answering: For each question the retriever returns the top 4 chunks (k=4). They are placed in a system prompt that tells the model to answer only from that context and to say it does not have the information instead of guessing. The model is openai/gpt-oss-20b served by Groq, with temperature 0.2 so answers stay factual. The interface is a Gradio ChatInterface and the pipeline uses LangChain.

RAG chatbot, problems solved: (1) Memory: embeddings with sentence-transformers and PyTorch exceeded Render's 512 MB limit; switching to FastEmbed (ONNX Runtime) reduced memory from over 512 MB to about 330 MB. (2) The Groq model llama-3.1-8b-instant was retired, so Nicolas listed the available models through the API and moved to openai/gpt-oss-20b. (3) fastembed 0.4.2 did not exist; version 0.7.3 was used. (4) A numpy version conflict on Python 3.12 was fixed by pinning Python 3.10.13.

RAG chatbot, indexing memory bug: When Nicolas expanded the knowledge base from a few chunks to about 80, the service started running out of memory on every start. The cause was the embedding library's default batch size of 256, which embedded everything in one giant batch: measured peak memory was 1064 MB for 76 chunks. Setting the batch size to 2 with one thread cut the peak to about 340 MB and startup still takes about 5 seconds.

RAG chatbot, limits: It can only answer from the documents in its data folder, so questions outside them get an honest "I don't have that information". It runs on Render's free tier, which sleeps when idle, so the first request after a pause can take a minute or more.
