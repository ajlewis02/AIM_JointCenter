import tensorflow as tf
import BuildModel
import numpy as np
import matplotlib.pyplot as plt
import random


def fixed_dist_loss():
    f = open(filename+".csv")
    flab = open(filename+"-labs.csv")
    losses = []
    for line in f:
        lineparsed = line.split("\n")[0].split(",")
        label = list(map(float, flab.readline().split("\n")[0].split(",")))
        dat = list(map(float, lineparsed))
        guess = model.predict([dat])
        losses.append(BuildModel.pythag_loss(np.array([label]), np.array(guess)).numpy()[0])
    print(sum(losses)/len(losses))
    print(max(losses))
    print(min(losses))
    print(len(losses))
    f.close()
    flab.close()


def normed_loss():
    """Norms the pythagorean distance by the distance from the joint center to the nearest marker. Only works on
    datasets which aren't already normed!"""
    f = open(filename+".csv")
    flab = open(filename+"-labs.csv")
    fdist = open(filename+"-dists.csv")

    losses = []
    for line in f:
        lineparsed = line.split("\n")[0].split(",")
        label = list(map(float, flab.readline().split("\n")[0].split(",")))
        dat = list(map(float, lineparsed))
        guess = model.predict([dat])
        dist = float(fdist.readline())
        losses.append(BuildModel.pythag_loss(np.array([label]), np.array(guess)).numpy()[0]/dist)
    print(sum(losses)/len(losses))
    print(max(losses))
    print(min(losses))
    print(len(losses))
    f.close()
    flab.close()
    fdist.close()


def simple_loss():
    f = open(filename+".csv")

    losses = []
    dists = []
    seq_len = 5
    for line in f:
        lineparsed = line.split("\n")[0].split(",")
        label = list(map(float, lineparsed[seq_len*12:(seq_len*12)+3]))
        dat = list(map(float, lineparsed[:seq_len*12]))
        dist = float(lineparsed[(seq_len*12)+3])
        guess = model.predict([dat])
        losses.append(BuildModel.pythag_loss(np.array([label]), np.array(guess)).numpy()[0])
        dists.append(dist)
    f.close()
    print(sum(losses)/len(losses))
    print(max(losses))
    print(min(losses))
    print(len(losses))
    print()
    distlosses = [losses[n]/dists[n] for n in range(len(losses))]
    print(sum(distlosses)/len(distlosses))
    print(max(distlosses))
    print(min(distlosses))
    print(len(distlosses))


def plot_timestep(timestep=-1):
    f = open(filename+".csv")
    flab = open(filename+"-labs.csv")

    labs = flab.readlines()
    if timestep == -1:
        timestep = random.randint(0, len(labs)-1)

    label = list(map(float, labs[timestep].split("\n")[0].split(",")))
    dat = list(map(float, f.readlines()[timestep].split("\n")[0].split(",")))
    print(dat)

    guess = model.predict([dat])[0]
    xdat = [dat[n] for n in range(0, len(dat), 3)]
    ydat = [dat[n] for n in range(1, len(dat), 3)]
    zdat = [dat[n] for n in range(2, len(dat), 3)]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xdat, ydat, zdat, c="Blue")
    ax.scatter([label[0]], [label[1]], [label[2]], c="Green")
    ax.scatter([guess[0]], [guess[1]], [guess[2]], c="Red")
    print(guess)
    plt.show()
    f.close()
    flab.close()


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    model_name = "flat_len5_gen2"
    filename = "simple_knee_seq_hard_len5_flat_norm-test"

    custom_objects = {"pythag_loss_no_norm":BuildModel.pythag_loss_no_norm}

    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(model_name)

    simple_loss()
