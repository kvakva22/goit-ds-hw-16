import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

model_cnn = load_model("C:/Users/Arsenii/Desktop/project/cnn_model.h5")
model_vgg = load_model("C:/Users/Arsenii/Desktop/project/vgg16_model.h5")

st.title('Робота нейронної мережі')

option = st.selectbox('Оберіть модель', ['CNN', 'VGG16'])

uploaded_file = st.file_uploader('Завантажте зображення на якому зображена цифра', type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption = 'Завантажене зображення')
    if st.button('Почати роботу нейронної мережі'):


        if option == 'CNN':
            history = pd.read_csv('training.log1')
            img = img.resize((28, 28))
            img = ImageOps.grayscale(img)
            img_array = np.array(img) / 255
            img_array = img_array.reshape(1, 28, 28, 1)
            prediction = model_cnn.predict(img_array)
            predicted_class = np.argmax(prediction)
            st.write(f'Predicted value:{predicted_class}')
            loss = history['loss']
            accuracy = history['accuracy']

            epochs = range(1, len(accuracy) + 1)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(epochs, accuracy, "b", label="accuracy")
            ax.plot(epochs, loss, "r", label="loss")
            ax.set_title('Accuracy and loss')
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Value")
            ax.legend()
            st.pyplot(fig)

        elif option == 'VGG16':
            history = pd.read_csv('training.log2')
            img = img.resize((224, 224))
            img = img.convert('RGB')
            img_array = np.array(img) / 255
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model_vgg.predict(img_array)
            predicted_class = np.argmax(prediction)
            st.write(f'Predicted value:{predicted_class}')
            loss = history['loss']
            accuracy = history['accuracy']

            epochs = range(1, len(accuracy) + 1)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(epochs, accuracy, "b", label="accuracy")
            ax.plot(epochs, loss, "r", label="loss")
            ax.set_title('Accuracy and loss')
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Value")
            ax.legend()
            st.pyplot(fig)