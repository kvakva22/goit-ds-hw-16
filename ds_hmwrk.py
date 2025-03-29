import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import pandas as pd
from keras.datasets import mnist
from PIL import Image, ImageOps
from keras import layers
from keras.models import Sequential
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16


def model_conv():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)

    x_train, x_test = x_train.reshape(-1, 28, 28, 1), x_test.reshape(-1, 28, 28, 1)

    x_train, x_test = x_train/255, x_test/255

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10) 

    modelcon = Sequential([
        layers.Conv2D(32,(3, 3), activation ='relu', input_shape = (28, 28, 1)),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64,(3, 3), activation ='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128, (3,3), activation = 'relu'),
        layers.Flatten(),

        layers.Dense(128, activation = 'relu'),
        layers.Dense(10, activation="softmax")
    ])

    modelcon.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

    historyconv = modelcon.fit(x_train, y_train, batch_size = 100, epochs=5, validation_data =(x_test, y_test))
    
    return modelcon, historyconv



def model_vgg():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, y_train = x_train[:5000], y_train[:5000]
    x_test, y_test = x_test[:1000], y_test[:1000]

    x_train = np.repeat(x_train[..., np.newaxis], 3, axis=-1)
    x_test = np.repeat(x_test[..., np.newaxis], 3, axis=-1)

    def batch(images, batch_size):
        batches = [tf.image.resize(images[i:i+batch_size], (224, 224)) for i in range(0, len(images), batch_size)]
        return np.concatenate(batches, axis=0)

    x_train = batch(x_train[:35000], batch_size=10000)  
    x_test = batch(x_test[:10000], batch_size=10000)

    x_train, x_test = x_train / 255.0, x_test / 255.0

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10) 

    base = VGG16(weights ='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base.layers:
        layer.trainable = False

    modelvg = Sequential([
        base,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(10, activation='softmax') 
    ])

    modelvg.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    historyvg=modelvg.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_test, y_test))

    return modelvg, historyvg




st.title('Робота нейронної мережі')

option = st.selectbox('Оберіть модель', ['CNN', 'VGG16'])

uploaded_file = st.file_uploader('Завантажте зображення на якому зображена цифра', type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption = 'Завантажене зображення')
    if st.button('Почати роботу нейронної мережі'):


        if option == 'CNN':
            modelconv, historyconv = model_conv()
            
            img = img.resize((28, 28))
            img = ImageOps.grayscale(img)
            img_array = np.array(img) / 255
            img_array = img_array.reshape(1, 28, 28, 1)
            prediction = modelconv.predict(img_array)
            predicted_class = np.argmax(prediction)
            st.write(f'Predicted value:{predicted_class}')
            loss = historyconv.history['loss']
            accuracy = historyconv.history['accuracy']

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
            modelvg, historyvg = model_vgg()

            img = img.resize((224, 224))
            img = img.convert('RGB')
            img_array = np.array(img) / 255
            img_array = np.expand_dims(img_array, axis=0)
            prediction = modelvg.predict(img_array)
            predicted_class = np.argmax(prediction)
            st.write(f'Predicted value:{predicted_class}')
            loss = historyvg.history['loss']
            accuracy = historyvg.history['accuracy']

            epochs = range(1, len(accuracy) + 1)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(epochs, accuracy, "b", label="accuracy")
            ax.plot(epochs, loss, "r", label="loss")
            ax.set_title('Accuracy and loss')
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Value")
            ax.legend()
            st.pyplot(fig)