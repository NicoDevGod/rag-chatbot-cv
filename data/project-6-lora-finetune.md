# Project 6: LoRA fine-tuning

LoRA fine-tuning: Nicolas's sixth AI project fine-tunes distilgpt2 (82 million parameters) to write in pirate style using LoRA (Low-Rank Adaptation). Code: https://github.com/NicoDevGod/lora-finetune. Live demo: https://lora-finetune.onrender.com. The goal was to understand parameter-efficient fine-tuning and then deploy it within Render's 512 MB limit.

LoRA fine-tuning, the idea: Full fine-tuning updates all parameters and produces a whole copy of the model per task. LoRA freezes the original weight matrix W and learns a low-rank update delta-W = A x B with much smaller matrices; only A and B train. His adapter trains 811,008 parameters out of 82,723,584 (0.98%) and the saved adapter is only 3.2 MB versus about 330 MB for the full model.

LoRA fine-tuning, training: Settings are rank r=16, lora_alpha 32, target modules c_attn and c_proj, dropout 0.05, trained with a hand-written PyTorch loop (no Trainer) for 20 epochs at learning rate 5e-4. The data is a synthetic rule-based corpus of about 300 pirate-style sentences, so there are no facts to get wrong, only a style to learn. Loss fell from 3.94 to 0.23 and the style generalized to prompts outside the corpus.

LoRA fine-tuning, first attempt failed: The first run (4 epochs, learning rate 2e-4, LoRA only on c_attn, rank 8) lowered the loss from 7.66 to 2.74 but the model never produced pirate style. Raising the rank to 16, adding c_proj, increasing the learning rate and training for 20 epochs fixed it. For a small dataset and a strong style change, overfitting a little is better than falling short.

LoRA fine-tuning, deployment: The adapter is merged into distilgpt2 (merge_and_unload), exported to ONNX with optimum and quantized to INT8, shrinking each model from about 330 MB to about 82 MB. Production serves it with onnxruntime, the tokenizers library and NumPy only, using a hand-written autoregressive decode loop (KV-cache bookkeeping and top-p sampling), with no PyTorch, transformers or optimum.

LoRA fine-tuning, the deploy story: It took four deploy attempts on Render. PyTorch directly ran out of memory; ONNX with optimum also did, even though it measured about 425 MB on Windows. Logging memory inside Render's container showed 490.5 MB after loading the model and 505.6 MB at the start of a request, about 6 MB of headroom, because merely importing torch cost 100 to 200 MB. Removing torch entirely brought it to about 283 MB at startup and about 345 MB under repeated requests, stable over six consecutive generations.
