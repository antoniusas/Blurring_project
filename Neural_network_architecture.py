import numpy as np
import tensorflow as tf


custom_model = tf.keras.applications.NASNetLarge(input_shape=None,
                                                 include_top=True,
                                                 weights="imagenet",
                                                 input_tensor=None,
                                                 pooling=None,
                                                 classes=1000,)

SUCCESS = 1; FAILURE = -1; COMPLETE = 0;

"Implement the neural network architecture"

class CustomModel1(tf.keras.Model):
    def __init__(self):
        super(CustomModel1, self).__init__()
        self.conv3d_1 = tf.keras.layers.Conv3D(filters=25, kernel_size=5, strides=(3, 3, 3), padding='same', activation='relu')
        self.maxpool3D_1 = tf.keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=None, padding='valid')
        self.conv3d_2 = tf.keras.layers.Conv3D(filters=10, kernel_size=3, strides=(2, 2, 2), padding='same', activation='relu')
        self.maxpool3D_2 = tf.keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=None, padding='valid')
        self.conv3d_3 = tf.keras.layers.Conv3D(filters=5, kernel_size=2, strides=(1, 1, 1), padding='same', activation='relu')
        self.maxpool3D_3 = tf.keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=None, padding='valid')

        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(rate=0.5)
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(5, activation='linear')

    def call(self, inputs):
        # Convolutional Layers
        x = self.conv3d_1(inputs)
        x = self.maxpool3D_1(x)

        # Fully connected layers with dropout
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return self.dense3(x)

"PLACEHOLDER for neural network architecture and model"
class CustomModel2(tf.keras.Model):
    def __init__(self):
        super(CustomModel2,self).__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)


    def call(self,inputs):
        x = self.dense1(inputs)
        x = self.dropout(x)
        return self.dense2(x)

def main(_):
    model = CustomModel1()
    """
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    """

if __name__ == '__main__':
    main(None)
