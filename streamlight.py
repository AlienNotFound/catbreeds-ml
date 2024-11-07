
import os
import cv2
import numpy as np
import tensorflow as tf

import streamlit as st
from PIL import Image

# import program

import tempfile
### Streamlit App ###
model = tf.keras.models.load_model('model_of_cats.keras')
class_names = model.class_names if hasattr(model, "class_names") else []

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
    # image_np = preprocess_image(uploaded_image)
    #image_np = np.array(Image.open(uploaded_image))

    predictions = model.predict(processed_image)

    predicted_class = class_names[np.argmax(predictions)] #-if class_names else "Unknown"

    st.image(uploaded_image, caption = "Uploaded image", use_container_width = True)
    st.write(f"Predicted breed: {predicted_class}")
    # temp_image_path = os.path.join(tempfile.gettempdir(), 'temp_file.jpg')
    # cv2.imwrite(temp_image_path, image_np)

    # resize_image = resize_image(uploaded_image, (300, 300))

    # col1, col2, col3 = st.columns(3)
    # # Placing the image in the second column will ensure it is displayed in the center of the application.
    # with col2:
    #     st.image(resize_image)

    # # Here we make a button to predict the image
    # submit = st.button('Predict')

    # if submit:
    #     # Load model

    #     # Use model to predict new image
    #     prediction = classify_digit(model, temp_image_path)
    #     predicted_digit = np.argmax(prediction)
    #     confidence = prediction[0][predicted_digit]

    #     st.subheader('Prediction Result')

    #     # Using np.argmax(prediction) will reveal the number with the highest probability as predicted by our model
    #     # st.success(f'The digit is probably a {np.argmax(prediction)}')

    #     st.success(f'The cat is probably a {predicted_digit} with {confidence * 100:.2f}% confidence')

    #     st.bar_chart(prediction[0])
    
    # os.remove(temp_image_path)
