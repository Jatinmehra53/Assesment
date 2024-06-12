import tensorflow as tf
import tensorflow_hub as hub
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image
import requests
import numpy as np

# here I've used pre-trained models from kaggle 
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")


def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    return np.array(image)


def detect_objects(image):
    input_tensor = tf.convert_to_tensor(image, dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detector(input_tensor)
    return detections


def generate_caption(image):
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption


image_path = "C:/Users/mehra/OneDrive/Desktop/NewYork.jpg" 
image = load_image(image_path)


detections = detect_objects(image)
print("Detected objects:", detections)


caption = generate_caption(image)
print("Generated caption:", caption)
