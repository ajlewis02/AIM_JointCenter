import os

import tensorflow as tf
from matplotlib import animation

import BuildModel
import numpy as np
import matplotlib.pyplot as plt
import random
from Vicon.Mocap.Vicon import Vicon
from Vicon.Markers import Markers
from GaitCore import Core
import LoadData

exoFrames = {"back": [Core.Point.Point(0, 14, 0),
                      Core.Point.Point(56, 0, 0),
                      Core.Point.Point(14, 63, 0),
                      Core.Point.Point(56, 63, 0)],
             "Shank": [Core.Point.Point(0, 0, 0),
                       Core.Point.Point(0, 63, 0),
                       Core.Point.Point(70, 14, 0),
                       Core.Point.Point(35, 49, 0)],
             "Thigh": [Core.Point.Point(0, 0, 0),
                       Core.Point.Point(70, 0, 0),
                       Core.Point.Point(0, 42, 0),
                       Core.Point.Point(70, 56, 0)]}


def fixed_dist_loss(filename):
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


def normed_loss(filename):
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


def simple_loss(filename, seq_len):
    f = open(filename+".csv")

    losses = []
    dists = []
    for line in f:
        lineparsed = line.split("\n")[0].split(",")
        label = list(map(float, lineparsed[seq_len*12:(seq_len*12)+3]))
        dat = list(map(float, lineparsed[:seq_len*12]))
        dist = float(lineparsed[(seq_len*12)+3])
        guess = model.predict([dat])
        losses.append(BuildModel.pythag_loss(np.array([label]), np.array(guess)).numpy()[0])
        dists.append(dist)
    f.close()
    # print(sum(losses)/len(losses))
    # print(max(losses))
    # print(min(losses))
    # print(len(losses))
    # print()
    distlosses = [losses[n]/dists[n] for n in range(len(losses))]
    # print(sum(distlosses)/len(distlosses))
    # print(max(distlosses))
    # print(min(distlosses))
    # print(len(distlosses))
    return distlosses


def simple_loss_from_raw(filename, seq_len):
    f = open(filename)

    v = Vicon(filename)
    source = v.get_markers()
    source.smart_sort()
    source.auto_make_transform(exoFrames)
    pos = []
    distlosses = []
    for i in range(0, len(source.get_marker("knee_top")) - seq_len, 1):
        data = []
        label = []
        dist = 1
        for n in range(seq_len):
            tstep = i + n
            tstepdat = LoadData.child_by_parent(source, tstep, "Thigh", "Shank")  # Raw positional data

            label = LoadData.hard_exo_joint_by_parent(source, tstep, "Thigh")
            dist = LoadData.shortest_pythag(tstepdat, label)

            tstepdat = [j/500 for j in tstepdat]  # Scale data down
            label = [j/500 for j in label]  # Scale label down too
            dist /= 500

            data = data + tstepdat

        guess = model.predict([data])[0]
        guess = [n*500 for n in guess]
        guess = local_to_global(guess, "Thigh", i+seq_len-1, source)
        dist *= 500
        label = LoadData.flatten_point(source.get_marker("knee_top")[i])
        pos.append(guess)
        distlosses.append(BuildModel.pythag_loss(np.array([label]), np.array(guess)).numpy()[0]/dist)
    return distlosses


def plot_timestep(filename, timestep=-1):
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


def local_to_global(point, frame, timestep, source):
    f = source.get_frame(frame)[timestep]
    return LoadData.flatten_point(Markers.local_point_to_global(f, Core.Point.Point(float(point[0]), float(point[1]), float(point[2]))))


def global_body(source, timestep, body):
    outx, outy, outz = [], [], []
    dat = source.get_rigid_body(body)
    points = [dat[n][timestep] for n in range(4)]
    for point in points:
        g_point = LoadData.flatten_point(point)
        outx.append(g_point[0])
        outy.append(g_point[1])
        outz.append(g_point[2])
    return outx, outy, outz


def vis_data(seq_len, filename):
    v = Vicon(filename)
    source = v.get_markers()
    source.smart_sort()
    source.auto_make_transform(exoFrames)
    pos = []
    losses = []
    for i in range(0, len(source.get_marker("knee_top")) - seq_len, 1):
        data = []
        label = []
        dist = 1
        for n in range(seq_len):
            tstep = i + n
            tstepdat = LoadData.child_by_parent(source, tstep, "Thigh", "Shank")  # Raw positional data

            label = LoadData.hard_exo_joint_by_parent(source, tstep, "Thigh")
            dist = LoadData.shortest_pythag(tstepdat, label)

            tstepdat = [j/500 for j in tstepdat]  # Scale data down
            label = [j/500 for j in label]  # Scale label down too
            dist /= 500

            data = data + tstepdat

        guess = model.predict([data])[0]
        guess = [n*500 for n in guess]
        guess = local_to_global(guess, "Thigh", i+seq_len-1, source)
        label = LoadData.flatten_point(source.get_marker("knee_top")[i])
        pos.append(guess)
        losses.append(BuildModel.pythag_loss(np.array([label]), np.array(guess)).numpy()[0]/dist)

    # Animate
    body_x = []
    body_y = []
    body_z = []

    joint_x = []
    joint_y = []
    joint_z = []

    guess_x = [0 for n in range(0, seq_len-1)]
    guess_y = [0 for n in range(0, seq_len-1)]
    guess_z = [0 for n in range(0, seq_len-1)]
    for i in range(0, seq_len-1):  # First seq_len-1 timesteps won't have guesses associated with them!
        thigh = global_body(source, i, "Thigh")
        shank = global_body(source, i, "Shank")
        joint = LoadData.flatten_point(source.get_marker("knee_top")[i])

        body_x.append(thigh[0] + shank[0])
        body_y.append(thigh[1] + shank[1])
        body_z.append(thigh[2] + shank[2])

        joint_x.append(joint[0])
        joint_y.append(joint[1])
        joint_z.append(joint[2])
    for i in range(seq_len, len(source.get_marker("knee_top"))):  # Animate with guess added
        thigh = global_body(source, i, "Thigh")
        shank = global_body(source, i, "Shank")
        joint = LoadData.flatten_point(source.get_marker("knee_top")[i])

        body_x.append(thigh[0] + shank[0])
        body_y.append(thigh[1] + shank[1])
        body_z.append(thigh[2] + shank[2])

        joint_x.append(joint[0])
        joint_y.append(joint[1])
        joint_z.append(joint[2])

        guess_x.append(pos[i-seq_len][0])
        guess_y.append(pos[i-seq_len][1])
        guess_z.append(pos[i-seq_len][2])

    _fig = plt.figure()
    _ax = _fig.add_subplot(111, projection='3d')
    _ax.set_autoscale_on(False)

    ani = animation.FuncAnimation(_fig,
                                  _animate, len(source.get_marker("knee_top")),
                                  fargs=(_ax, [body_x, body_y, body_z], [joint_x, joint_y, joint_z], [guess_x, guess_y, guess_z]),
                                  interval=100 / 10)
    plt.show()


def _animate(frame, _ax, body, joint, guess=None):
    _ax.clear()
    _ax.set_xlabel('X')
    _ax.set_ylabel('Y')
    _ax.set_zlabel('Z')
    _ax.axis([-500, 500, -500, 500])
    _ax.set_zlim3d(0, 1250)
    _ax.scatter(body[0][frame], body[1][frame], body[2][frame], c='r', marker='o')
    _ax.scatter(joint[0][frame], joint[1][frame], joint[2][frame], c="b", marker="o")
    _ax.scatter(guess[0][frame], guess[1][frame], guess[2][frame], c="g", marker="o")


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    model_name = "flat_len5_addl_norm_gen2"

    custom_objects = {"pythag_loss_no_norm":BuildModel.pythag_loss_no_norm}

    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(model_name)

    val = simple_loss("simple_knee_seq_hard_len5_flat_norm_addl-val", 5)
    print("Validation")
    train = simple_loss("simple_knee_seq_hard_len5_flat_norm_addl", 5)
    print("Training")
    test = simple_loss("simple_knee_seq_hard_len5_flat_norm_addl-test", 5)
    print("Test1")
    test2 = simple_loss("simple_knee_seq_hard_len5_flat_norm-test", 5)

    plt.boxplot([val, train, test, test2], labels=["Validation Data", "Training Data", "Testing Data", "Testing Data2"], showfliers=False)
    plt.ylabel("Relative Error")
    plt.show()

    # exo_sources = ["./Sources/" + n for n in os.listdir("./Sources")]
    #
    # exo_val = ["./ValSources/" + n for n in os.listdir("./ValSources")]
    #
    # exo_test = ["./TestSources/" + n for n in os.listdir("./TestSources")]
    #
    # exo_all = exo_val+exo_sources+exo_test
    #
    # for i in range(len(exo_all)):
    #     print("Median of " + exo_all[i] + ": " + str(np.median(simple_loss_from_raw(exo_all[i], 5))))
    # vis_data(5, "./TestSources/bent_neg_yplane01.csv")
