
import cv2
import numpy as np
import tensorflow as tf
import json
import streamlit as st
from PIL import Image

### Streamlit App ###
model = tf.keras.models.load_model('model_of_cats_short.keras')
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

## Classify image
def classify_digit(model, image):
    img = cv2.imread(image)[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    
    return prediction

## Resize image
def resize_image(image, target_size):
    img = Image.open(image)
    resize_image = img.resize(target_size)

    return resize_image

def preprocess_image(uploaded_image, img_height = 180, img_width = 180):
    image = Image.open(uploaded_image)

    image = image.convert("RGB")
    image = image.resize((img_width, img_height))
    image_array = np.array(image)

    image_array = np.expand_dims(image_array, axis = 0)
    return image_array

# Page name
st.set_page_config('Digit Recognition', page_icon = '#')

st.title('# Cat Breed Recognition #')


uploaded_image = st.file_uploader('Insert a picture of a cat', type = 'jpg')

if uploaded_image is not None:
    processed_image = preprocess_image(uploaded_image)

    predictions = model.predict(processed_image)

    predicted_class = class_names[np.argmax(predictions)] if class_names else "Unknown"

    st.image(uploaded_image, caption = "Uploaded image", use_container_width = True)
    st.write(f"Predicted breed: {predicted_class}")