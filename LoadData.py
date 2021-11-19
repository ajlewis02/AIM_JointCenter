import numpy as np, math
from GaitCore import Core
import pandas as pd
import matplotlib.pyplot as plt
from Vicon.Mocap.Vicon import Vicon
from Vicon.Markers import Markers
import random, os
from scipy.spatial.transform import Rotation as R

#  TODO: verify exoskeleton frames
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

markers = []
markersVal = []


def setup_exo_mocap(sources):
    """Sets up the exoskeleton data, using the mocap-calculated joint centers"""
    for source in sources:
        v = Vicon(source)
        markers.append(v.get_markers())

    for m in markers:
        m.smart_sort()
        m.auto_make_transform(exoFrames)
        for j in [["knee", "thigh", "shank", False]]:
            m.def_joint(j[0], j[1], j[2], j[3])
        m.calc_joints(verbose=True)
        m.save_joints()


def setup_exo_hard(sources, val):
    """Sets up the exoskeleton data without calculating the joint center"""
    for source in sources:
        v = Vicon(source)
        markers.append(v.get_markers())

    for m in markers:
        m.smart_sort()
        m.auto_make_transform(exoFrames)

    for source in val:
        v = Vicon(source)
        markersVal.append(v.get_markers())

    for m in markersVal:
        m.smart_sort()
        m.auto_make_transform(exoFrames)


def flatten_point(point):
    return [point.x, point.y, point.z]


def child_by_parent(source, timestep, parent, child):
    """Returns a single timestep of the child rigid body's markers, in the parent rigid body's reference frame,
    using the specified markers object. Output is in format [x1, y1, z1, ... x4, y4, z4]"""
    parent_frame = source.get_frame(parent)[timestep]
    child_dat = source.get_rigid_body(child)
    child_points = [Markers.global_point_to_frame(parent_frame, child_dat[n][timestep]) for n in range(4)]
    out = []
    for point in child_points:
        out += flatten_point(point)
    return out


def hard_exo_joint_by_parent(source, timestep, parent):
    """Returns the location of the "knee" marker relative to the parent rigid body. Output is in format [x, y, z]"""
    parent_frame = source.get_frame(parent)[timestep]
    jc = source.get_marker("knee_top")[timestep]
    jc_rel = Markers.global_point_to_frame(parent_frame, jc)
    return flatten_point(jc_rel)


def pythag(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)


def shortest_pythag(timestep, label):
    dist = -1
    for i in range(0, len(timestep)-4, 3):
        newdist = pythag([timestep[i], timestep[i+1], timestep[i+2]], label)
        if dist < 0 or newdist < dist:
            dist = newdist
    return dist


def simple_seq_knee_hard(seq_len, filename, val=False):
    """Generator which returns a sequence of seq_len of sequential timesteps. Data is in format [child_by_parent(
    source, 1), child_by_parent(source, 2), ...], hard_exo_joint_by_parent()"""
    if val:
        datused = markersVal
        f = open(filename+"-val.csv", "w")
        flab = open(filename+"-val-labs.csv", "w")
    else:
        datused = markers
        f = open(filename+".csv", "w")
        flab = open(filename+"-labs.csv", "w")

    for source in datused:
        for i in range(0, len(source.get_marker("knee")) - seq_len, seq_len):
            data = []
            label = []
            for n in range(seq_len):
                tstep = i + n
                tstepdat = child_by_parent(source, tstep, "Thigh", "Shank")  # Raw positional data

                label = hard_exo_joint_by_parent(source, tstep, "Thigh")
                dist = shortest_pythag(tstepdat, label)

                for point in range(0, len(tstepdat)-2, 3):  # Transform data so that joint center would be at origin
                    tstepdat[point] -= label[0]
                    tstepdat[point+1] -= label[1]
                    tstepdat[point+2] -= label[2]

                tstepdat = [j/dist for j in tstepdat]  # Scale data so that nearest point to joint center is 1 dist away

                for point in range(0, len(tstepdat)-2, 3):  # Move points back
                    tstepdat[point] += label[0]
                    tstepdat[point+1] += label[1]
                    tstepdat[point+2] += label[2]

                data.append(tstepdat)
            for timestep in data:
                f.write(",".join([str(n) for n in timestep])+"\n")
            flab.write(",".join([str(n) for n in label])+"\n")
    f.close()
    flab.close()


def simple_seq_knee_hard_simple_norm(seq_len, filename, val=False):
    """Generator which returns a sequence of seq_len of sequential timesteps. Data is in format [child_by_parent(
    source, 1), child_by_parent(source, 2), ...], hard_exo_joint_by_parent()"""
    if val:
        datused = exo_val
        f = open(filename+"-val.csv", "w")
        flab = open(filename+"-val-labs.csv", "w")
    else:
        datused = exo_sources
        f = open(filename+".csv", "w")
        flab = open(filename+"-labs.csv", "w")

    for ss in datused:
        v = Vicon(ss)
        source = v.get_markers()
        source.smart_sort()
        source.auto_make_transform(exoFrames)
        for i in range(0, len(source.get_marker("knee_top")) - seq_len, 1):
            data = []
            label = []
            xnoise = np.random.normal(0, 65)
            ynoise = np.random.normal(0, 65)
            znoise = np.random.normal(0, 65)
            for n in range(seq_len):
                tstep = i + n
                tstepdat = child_by_parent(source, tstep, "Thigh", "Shank")  # Raw positional data

                label = hard_exo_joint_by_parent(source, tstep, "Thigh")
                dist = shortest_pythag(tstepdat, label)

                for j in range(0, len(tstepdat), 3):
                    tstepdat[j] += xnoise
                for j in range(1, len(tstepdat), 3):
                    tstepdat[j] += ynoise
                for j in range(2, len(tstepdat), 3):
                    tstepdat[j] += znoise

                label[0] += xnoise
                label[1] += ynoise
                label[2] += znoise

                tstepdat = [j/dist for j in tstepdat]  # Scale data so that nearest point to joint center is 1 dist away
                label = [j/dist for j in label]  # Scale label down too

                data = data + tstepdat
            f.write(",".join([str(n) for n in data])+"\n")
            flab.write(",".join([str(n) for n in label])+"\n")
    f.close()
    flab.close()


def simple_seq_knee_hard_no_norm(seq_len, filename, val=False):
    """Generator which returns a sequence of seq_len of sequential timesteps. Data is in format [child_by_parent(
    source, 1), child_by_parent(source, 2), ...], [hard_exo_joint_by_parent(), shortest_pythag]"""
    if val:
        datused = exo_val
        f = open(filename+"-val.csv", "w")
        flab = open(filename+"-val-labs.csv", "w")
    else:
        datused = exo_sources
        f = open(filename+".csv", "w")
        flab = open(filename+"-labs.csv", "w")

    for ss in datused:
        v = Vicon(ss)
        source = v.get_markers()
        source.smart_sort()
        source.auto_make_transform(exoFrames)
        for i in range(0, len(source.get_marker("knee_top")) - seq_len, 1):
            data = []
            label = []
            dist = 1
            xnoise = np.random.normal(0, 65)
            ynoise = np.random.normal(0, 65)
            znoise = np.random.normal(0, 65)
            for n in range(seq_len):
                tstep = i + n
                tstepdat = child_by_parent(source, tstep, "Thigh", "Shank")  # Raw positional data

                label = hard_exo_joint_by_parent(source, tstep, "Thigh")
                dist = shortest_pythag(tstepdat, label)

                for j in range(0, len(tstepdat), 3):
                    tstepdat[j] += xnoise
                for j in range(1, len(tstepdat), 3):
                    tstepdat[j] += ynoise
                for j in range(2, len(tstepdat), 3):
                    tstepdat[j] += znoise

                label[0] += xnoise
                label[1] += ynoise
                label[2] += znoise

                data = data + tstepdat
            f.write(",".join([str(n) for n in data])+"\n")
            flab.write(",".join([str(n) for n in label])+","+str(dist)+"\n")
    f.close()
    flab.close()


def simple_seq_knee_hard_flat_norm_centroid(seq_len, filename, datused, addl_noise=False, repeat=1):
    """Generator which returns a sequence of seq_len of sequential timesteps. Data is in format [child_by_parent(
    source, 1), child_by_parent(source, 2), ...], [hard_exo_joint_by_parent(), shortest_pythag]
    If addl_noise = True, the data will also be randomly scaled and rotated about the joint center"""
    f = open(filename, "w")
    for r in range(repeat):
        for ss in datused:
            v = Vicon(ss)
            source = v.get_markers()
            source.smart_sort()
            source.auto_make_transform(exoFrames)
            for i in range(0, len(source.get_marker("knee_top")) - seq_len, 1):
                data = []
                label = []
                dist = 1
                xnoise = np.random.normal(0, 65)
                ynoise = np.random.normal(0, 65)
                znoise = np.random.normal(0, 65)
                scale_noise = np.random.uniform(0.5, 1.5)
                rot_x = np.random.normal(0, math.pi/2)
                rot_y = np.random.normal(0, math.pi/2)
                rot_z = np.random.normal(0, math.pi/2)
                for n in range(seq_len):
                    tstep = i + n
                    tstepdat = child_by_parent(source, tstep, "Thigh", "Shank")  # Raw positional data

                    # Find the centroid
                    dat_x = [tstepdat[m] for m in range(0, len(tstepdat), 3)]
                    dat_y = [tstepdat[m] for m in range(1, len(tstepdat), 3)]
                    dat_z = [tstepdat[m] for m in range(2, len(tstepdat), 3)]

                    centroid = [sum(dat_x)/len(dat_x), sum(dat_y)/len(dat_y), sum(dat_z)/len(dat_z)]

                    label = hard_exo_joint_by_parent(source, tstep, "Thigh")
                    dist = pythag(centroid, label)

                    centroid[0] += xnoise
                    centroid[1] += ynoise
                    centroid[2] += znoise

                    label[0] += xnoise
                    label[1] += ynoise
                    label[2] += znoise

                    centroid = [j/500 for j in centroid]  # Scale data down
                    label = [j/500 for j in label]  # Scale label down too
                    dist /= 500

                    if addl_noise:
                        # Translate centroid so that label is at origin
                        centroid = [centroid[m] - label[m] for m in range(len(label))]

                        # Scale centroid (and dist!) about the origin
                        centroid = [m * scale_noise for m in centroid]
                        dist *= scale_noise

                        # Rotate centroid about origin
                        r = R.from_rotvec([rot_x, rot_y, rot_z])
                        vectors = [[centroid[m], centroid[m+1], centroid[m+2]] for m in range(0, len(centroid)-1, 3)]
                        rot_vectors = r.apply(vectors)
                        centroid = []
                        for j in rot_vectors:
                            for m in j:
                                centroid.append(m)

                        # Translate centroid back from origin
                        centroid = [centroid[m] + label[m] for m in range(len(label))]

                    data = data + centroid

                f.write(",".join([str(n) for n in data])+","+",".join([str(n) for n in label])+","+str(dist)+"\n")
    f.close()


if __name__ == "__main__":

    exo_sources = ["./Sources/" + n for n in os.listdir("./Sources")]

    exo_val = ["./ValSources/" + n for n in os.listdir("./ValSources")]

    exo_test = ["./TestSources/" + n for n in os.listdir("./TestSources")]

    simple_seq_knee_hard_flat_norm_centroid(10, "simple_knee_seq_hard_len10_flat_norm_centroid.csv", exo_sources)
    simple_seq_knee_hard_flat_norm_centroid(10, "simple_knee_seq_hard_len10_flat_norm_centroid-val.csv", exo_val)
    simple_seq_knee_hard_flat_norm_centroid(10, "simple_knee_seq_hard_len10_flat_norm_centroid-test.csv", exo_test)
