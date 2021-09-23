import tensorflow as tf
import numpy, random, math
from GaitCore import Core
import pandas as pd
import matplotlib.pyplot as plt
from Vicon.Mocap.Vicon import Vicon
from Vicon.Markers import Markers
import LoadData


def pythag_loss(y_true, y_pred):
        pythag = tf.norm(y_pred - y_true, axis=1)
        return pythag


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    seq_len = 1
    filename = "simple_knee_seq_hard_len1_simple_norm"

    train_ds = pd.read_csv(filename+".csv")
    val_ds = pd.read_csv(filename+"-val.csv")

    train_ds_labels = pd.read_csv(filename+"-labs.csv")
    val_ds_labels = pd.read_csv(filename+"-val-labs.csv")

    model_name = "TEST2"

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(12 * seq_len, activation="tanh"),
        tf.keras.layers.Dense(3)
    ], model_name)

    model.compile(optimizer=tf.optimizers.Adam(0.001), loss=pythag_loss)
    history = model.fit(train_ds, train_ds_labels, epochs=100, verbose=0, validation_data=(val_ds, val_ds_labels))

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)

    plt.show()

    model.save(model_name)
