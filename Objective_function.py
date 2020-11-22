import numpy as np
import tensorflow as tf

"!!!-- Remember to use tensorflow(tf) equivalent of numpy(np) functions in order to maintain gradient information --!!!"
SUCCESS = 1; FAILURE = -1; COMPLETE = 0;


"Create loss function classification"
def loss_fnc_person_classifier(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

"Create loss function bounding_box "
def loss_fnc_bounding_box(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

"Create loss function reconstruction_error"
def loss_reconstruction_error(y_true, y_pred):
    return COMPLETE

def


def main(_):
    return COMPLETE

if __name__ == '__main__':
    main(None)
