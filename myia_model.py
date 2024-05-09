import tensorflow as tf
from tensorflow.keras import layers, models


class Myia(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # model trains if below line input_shape=(150, 200, 3) ????, rbg and alpha layer?
        self.conv2d1 = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 200, 3))
        self.maxpool1 = layers.MaxPooling2D((2, 2))
        self.conv2d2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.maxpool2 = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        # self.dense1 = *[layers.Dense(64, activation='relu') for _ in range(1)]  # Add dense layers based on config
        self.dense2 = layers.Dense(1, activation='sigmoid')
            
    def call(self, x):
        x = self.conv2d1(x)
        x = self.maxpool1(x)
        x = self.conv2d2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        # x = self.dense1(x)
        x = self.dense2(x)
        return x 