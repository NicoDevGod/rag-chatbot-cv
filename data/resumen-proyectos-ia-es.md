# Resumen en español de los proyectos de IA de Nicolas

Resumen de proyectos de IA: Nicolas Alejandro Angel Robles construyó seis proyectos de IA en Python, todos desplegados en Render (plan gratuito, 512 MB de RAM) con una demo en vivo hecha en Gradio, código público en GitHub (NicoDevGod) y una explicación paso a paso en docs/HOW_IT_WORKS.md. Su portafolio de IA está en https://nicodevgod.github.io/ai-portfolio/ y su portafolio anterior de desarrollo web en https://porfolionicodev.netlify.app/.

Demos en vivo de los proyectos de IA: (1) chatbot RAG https://rag-chatbot-cv-zre6.onrender.com, (2) clasificador de residuos https://waste-classifier-keon.onrender.com, (3) agente con herramientas https://tool-agent-jgfi.onrender.com, (4) recomendador de películas https://movie-recommender-ej1u.onrender.com, (5) detector de objetos https://object-detector-rph8.onrender.com, (6) fine-tuning con LoRA https://lora-finetune.onrender.com. El plan gratuito de Render duerme cuando no se usa, así que una demo puede tardar hasta dos minutos en despertar.

Proyecto 1, chatbot RAG: es este chatbot. Responde preguntas sobre la experiencia de Nicolas recuperando fragmentos de sus propios documentos (chunks de 800 caracteres, embeddings all-MiniLM-L6-v2 con FastEmbed, Chroma, 4 fragmentos por pregunta) y generando la respuesta con Groq (openai/gpt-oss-20b). Pasó de más de 512 MB a unos 330 MB de RAM al reemplazar PyTorch por FastEmbed (ONNX).

Proyecto 2, clasificador de residuos: clasifica una foto en cartón, vidrio, metal, papel, plástico o basura con transfer learning sobre MobileNetV2 (TrashNet, 150 imágenes por clase, unos 78% de accuracy tras 5 épocas). El modelo se exporta a ONNX (8,9 MB) y se sirve sin PyTorch.

Proyecto 3, agente con herramientas: un agente sin frameworks sobre la API de function calling de Groq, con cuatro herramientas: calculadora segura (ast, nunca eval), clima (Open-Meteo), Wikipedia y hora actual. Tiene un máximo de 5 pasos por pregunta.

Proyecto 4, recomendador de películas: filtrado colaborativo con MovieLens usando una matriz rala y similitud coseno, sin redes neuronales. Calcula la similitud por consulta en vez de precalcular toda la matriz, y fue el único proyecto que se desplegó sin bugs.

Proyecto 5, detector de objetos: YOLOv8n (80 clases de COCO) exportado a ONNX, con preprocesamiento letterbox, decodificación de las 8.400 cajas candidatas y Non-Max Suppression escritos a mano en NumPy, sin torch en producción.

Proyecto 6, fine-tuning con LoRA: distilgpt2 aprende estilo pirata entrenando solo el 0,98% de sus parámetros (adaptador de 3,2 MB). Para desplegarlo en 512 MB de RAM necesitó cuatro intentos: eliminó torch por completo y escribió a mano el loop de generación sobre onnxruntime (unos 283 MB al iniciar y unos 345 MB bajo uso).
