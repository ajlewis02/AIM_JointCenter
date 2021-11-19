import tensorflow as tf
import numpy as np, random, math
from GaitCore import Core
import pandas as pd
import matplotlib.pyplot as plt
from Vicon.Mocap.Vicon import Vicon
from Vicon.Markers import Markers
import LoadData


def pythag_loss(y_true, y_pred):
        pythag = tf.norm(y_pred - y_true, axis=1)
        return pythag


def pythag_loss_no_norm(y_true, y_pred):
    pythag = tf.norm(y_pred - y_true[:, :3], axis=1)
    return pythag/y_true[:, 3:]


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    seq_len = 10
    filename = "simple_knee_seq_hard_len10_flat_norm_centroid"

    train_ds_all = pd.read_csv(filename+".csv")
    val_ds_all = pd.read_csv(filename+"-val.csv")

    print(train_ds_all.shape)

    train_ds = train_ds_all.iloc[:, :seq_len*3]
    train_ds_labels = train_ds_all.iloc[:, seq_len*3:]
    val_ds = val_ds_all.iloc[:, :seq_len*3]
    val_ds_labels = val_ds_all.iloc[:, seq_len*3:]

    train_ds_tf = tf.data.Dataset.from_tensor_slices((train_ds, train_ds_labels)).shuffle(1000).batch(100)
    val_ds_tf = tf.data.Dataset.from_tensor_slices((val_ds, val_ds_labels)).shuffle(1000).batch(100)

    # train_ds_labels = pd.read_csv(filename+"-labs.csv")
    # val_ds_labels = pd.read_csv(filename+"-val-labs.csv")

    model_name = "flat_len10_norm_centroid_gen1"

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(3 * seq_len, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
        tf.keras.layers.Dense(3 * seq_len, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
        tf.keras.layers.Dense(3 * seq_len, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
        tf.keras.layers.Dense(3 * seq_len, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
        tf.keras.layers.Dense(3 * seq_len, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
        tf.keras.layers.Dense(3)
    ], model_name)

    model.compile(optimizer=tf.optimizers.Adam(0.001), loss=pythag_loss_no_norm)
    history = model.fit(train_ds_tf, epochs=200, verbose=0, validation_data=val_ds_tf)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)

    print(history.history["loss"][len(history.history["loss"])-1])
    print(history.history['val_loss'][len(history.history['val_loss'])-1])

    plt.show()

    model.save(model_name)
