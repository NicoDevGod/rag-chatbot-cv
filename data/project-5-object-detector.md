# Project 5: Object detector

Object detector: Nicolas's fifth AI project detects objects in a photo with YOLOv8n and draws a box and a label around each one, for the 80 categories of the COCO dataset (people, vehicles, animals, furniture and more). Code: https://github.com/NicoDevGod/object-detector. Live demo: https://object-detector-rph8.onrender.com.

Object detector, classification vs detection: The waste classifier answers "what is this?" once per image. Detection answers what is in the image and where each thing is, and a photo can hold zero, one or many objects. YOLOv8n is pretrained on COCO; he exported it to ONNX (about 12 MB, with Ultralytics) so production runs only onnxruntime, NumPy and Pillow, with no PyTorch or torchvision.

Object detector, letterbox: YOLO expects a 640x640 input. Stretching a rectangular photo to a square distorts proportions and miscalibrates the boxes, so the image is scaled keeping its aspect ratio and padded with gray (114,114,114), the same trick video players use. The scale ratio and padding are saved to map boxes back to the original photo coordinates later.

Object detector, decoding: The model output has shape (1, 84, 8400): 8,400 candidate boxes, each with 4 box numbers (center x, center y, width, height) and 80 class probabilities. Almost all candidates are noise, so a confidence threshold of 35% keeps only a handful. Box coordinates are then converted back from the padded 640x640 space to the original image.

Object detector, NMS by hand: One object produces several overlapping boxes. Non-Max Suppression keeps the most confident one and discards others whose IoU (intersection over union) with it is above 0.45. He wrote IoU and NMS in plain NumPy instead of using torchvision.ops.nms, which would have required installing all of PyTorch for one function. A per-class offset trick keeps a dog from suppressing a neighboring cat.

Object detector, result: On a sample photo of a bus it detected the bus (84%) and four people (90%, 88%, 87% and 43%), identical locally and in production. There were no production bugs; the main decision was keeping the server lightweight by avoiding torch.
