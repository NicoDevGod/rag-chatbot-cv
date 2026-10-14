# Nicolas's AI and Python projects: overview

AI portfolio: Nicolas Alejandro Angel Robles has built six deployed AI projects in Python as a hands-on way to learn AI systems end to end. They are: (1) this RAG chatbot, (2) a waste classifier, (3) a tool-use agent, (4) a movie recommender, (5) an object detector and (6) a LoRA fine-tuning project.

AI portfolio, live demos: (1) RAG chatbot https://rag-chatbot-cv-zre6.onrender.com, (2) waste classifier https://waste-classifier-keon.onrender.com, (3) tool-use agent https://tool-agent-jgfi.onrender.com, (4) movie recommender https://movie-recommender-ej1u.onrender.com, (5) object detector https://object-detector-rph8.onrender.com, (6) LoRA fine-tuning https://lora-finetune.onrender.com. The free tier sleeps when idle, so a demo can take up to about two minutes to wake up.

AI portfolio: Every AI project follows the same cycle: build it, test it locally, publish the code on GitHub (user NicoDevGod), write a step-by-step explanation in docs/HOW_IT_WORKS.md, deploy it on Render's free tier (512 MB of RAM) and verify it against the live server. Each one has a public live demo built with Gradio.

AI portfolio: Nicolas's AI portfolio website is at https://nicodevgod.github.io/ai-portfolio/ (bilingual Spanish and English, with a case study, a live demo and a scaling roadmap for each project). His earlier web development portfolio is at https://porfolionicodev.netlify.app/. Contact details are in his profile: GitHub https://github.com/NicoDevGod and LinkedIn https://www.linkedin.com/in/nicolas-angel-b140b92a6/.

AI portfolio: Nicolas also wrote a 54-slide study deck (in Spanish) that explains the six projects in depth: architecture, concepts explained from scratch, the real bugs found on the way to production and how they were fixed. It is downloadable from the AI portfolio website.

AI portfolio: Technologies across the AI projects: Python, LangChain, Chroma, FastEmbed, Groq API, PyTorch, torchvision, MobileNetV2, YOLOv8, LoRA with peft, ONNX and ONNX Runtime, pandas, SciPy, scikit-learn, NumPy, Hugging Face datasets, and Gradio for the interfaces. Deployment is on Render with render.yaml Blueprints.

AI portfolio: A recurring theme is keeping production lightweight: models are trained or exported with heavy libraries on his own machine, but the deployed servers use small runtimes (ONNX Runtime, FastEmbed, NumPy) instead of PyTorch, so everything fits in 512 MB of RAM. Several projects reimplement a small function by hand instead of installing a whole library.

AI portfolio, hardest deployments and bugs: The project that was hardest to deploy was project 6 (LoRA fine-tuning), which needed four deploy attempts on Render because importing PyTorch alone used 100 to 200 MB of the 512 MB limit; he fixed it by removing torch completely. The RAG chatbot (project 1) also ran out of memory until PyTorch embeddings were replaced with FastEmbed. The movie recommender (project 4) and the object detector (project 5) deployed without production bugs.
