# Project 2: Waste classifier

Waste classifier: Nicolas's second AI project is a computer vision app that classifies a photo of waste into 6 recycling categories: cardboard, glass, metal, paper, plastic and trash. Code: https://github.com/NicoDevGod/waste-classifier. Live demo: https://waste-classifier-keon.onrender.com (the interface is in Spanish).

Waste classifier, approach: It uses transfer learning. Training an image network from scratch needs millions of labeled images and days of GPU time, so he started from MobileNetV2 pretrained on ImageNet (1.2 million photos), froze its convolutional layers (model.features) and replaced the last layer (classifier[1]) with a 6-output layer. Only that layer is trained, with horizontal-flip data augmentation.

Waste classifier, data and results: The dataset is TrashNet (garythung/trashnet on Hugging Face), which is heavily imbalanced, so he sampled 150 images per class (900 in total). After 5 epochs training accuracy went from about 50% to about 78%. In production tests: a cardboard photo scored 99.6%, glass 76.7% and paper 94.0% confidence.

Waste classifier, deployment: The trained model is exported to ONNX (8.9 MB) and served with ONNX Runtime, so production never installs PyTorch. Preprocessing must match training exactly: 160x160 resize, divide by 255, ImageNet mean and standard deviation normalization, and channel-first axis order. A mismatch does not crash; the model silently predicts worse. Softmax converts the 6 raw outputs into probabilities shown in a Gradio label.

Waste classifier, problems solved: (1) Streaming the dataset with Hugging Face datasets raised MemoryError on certain images; he switched to a normal download with manual balanced sampling. (2) gradio 5.7.1 crashed while building the API schema for gr.Label; upgrading to gradio 5.50.0 fixed it. (3) The first request after the free-tier server sleeps can be slow, which looked like a failure until the logs showed it was just a cold start.

Waste classifier, UI: The interface labels are translated to Spanish (cartón, vidrio, metal, papel, plástico, basura) only at display time; the model's label order stays in English because it drives the ONNX output indexes.
