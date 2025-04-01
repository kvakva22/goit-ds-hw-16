import tensorflow as tf
from keras.datasets import mnist
from keras import layers
from keras.models import Sequential
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.callbacks import CSVLogger
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)

x_train, x_test = x_train.reshape(-1, 28, 28, 1), x_test.reshape(-1, 28, 28, 1)

x_train, x_test = x_train/255, x_test/255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10) 

csv_logger1 = CSVLogger('training.log1', separator=',', append=False)
csv_logger2 = CSVLogger('training.log2', separator=',', append=False)

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

modelcon.fit(x_train, y_train, batch_size = 100, epochs=5, validation_data =(x_test, y_test), callbacks=[csv_logger1])
modelcon.save('cnn_model.h5')    


x_train, y_train = x_train[:5000], y_train[:5000]
x_test, y_test = x_test[:1000], y_test[:1000]

x_train = np.repeat(x_train, 3, axis=-1) 
x_test = np.repeat(x_test, 3, axis=-1)   


def batch(images):
    images = tf.cast(images, tf.float32)
    return tf.image.resize(images, (224, 224))

x_train = batch(x_train)  
x_test = batch(x_test) 


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

modelvg.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_test, y_test), callbacks=[csv_logger2])
modelvg.save('vgg16_model.h5')
