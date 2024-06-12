import tensorflow as tf
import tensorflow_hub as hub
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image
import requests
import numpy as np

# Load pre-trained object detection model (YOLO)
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# Load pre-trained image captioning model
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Function to load and preprocess image
def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    return np.array(image)

# Function to perform object detection
def detect_objects(image):
    input_tensor = tf.convert_to_tensor(image, dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detector(input_tensor)
    return detections

# Function to generate image caption
def generate_caption(image):
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

# Load image
image_path = "C:/Users/mehra/OneDrive/Desktop/NewYork.jpg"  # Change this to your image path
image = load_image(image_path)


# Detect objects in the image
detections = detect_objects(image)
print("Detected objects:", detections)

# Generate caption for the image
caption = generate_caption(image)
print("Generated caption:", caption)
