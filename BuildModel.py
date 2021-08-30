import tensorflow as tf
import numpy, random, math
from GaitCore import Core
import pandas as pd
import matplotlib as plt
from Vicon.Mocap.Vicon import Vicon
from Vicon.Markers import Markers

#  TODO: verify exoskeleton frames
exoFrames = {"back": [Core.Point.Point(0, 14, 0),
                      Core.Point.Point(56, 0, 0),
                      Core.Point.Point(14, 63, 0),
                      Core.Point.Point(56, 63, 0)],
             "shank": [Core.Point.Point(0, 0, 0),
                       Core.Point.Point(0, 63, 0),
                       Core.Point.Point(70, 14, 0),
                       Core.Point.Point(35, 49, 0)],
             "thigh": [Core.Point.Point(0, 0, 0),
                       Core.Point.Point(70, 0, 0),
                       Core.Point.Point(0, 42, 0),
                       Core.Point.Point(70, 56, 0)]}

markers = []


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


def setup_exo_hard(sources):
    """Sets up the exoskeleton data without calculating the joint center"""
    for source in sources:
        v = Vicon(source)
        markers.append(v.get_markers())

    for m in markers:
        m.smart_sort()
        m.auto_make_transform(exoFrames)


def flatten_point(point):
    return [point.x, point.y, point.z]


def child_by_parent(source, timestep, parent, child):
    """Returns a single timestep of the child rigid body's markers, in the parent rigid body's reference frame,
    using the specified markers object. Output is in format [x1, y1, z1, ... x4, y4, z4]"""
    parent_frame = source.get_frame(parent)[timestep]
    child_dat = source.get_rigid_body(child)
    child_points = [Markers.global_point_to_frame(parent_frame, child_dat[n][0]) for n in range(4)]
    out = []
    for point in child_points:
        out += flatten_point(point)
    return out


def hard_exo_joint_by_parent(source, timestep, parent):
    """Returns the location of the "knee" marker relative to the parent rigid body. Output is in format [x, y, z]"""
    parent_frame = source.get_frame(parent)[timestep]
    jc = source.get_marker("knee")
    jc_rel = Markers.global_point_to_frame(parent_frame, jc)
    return flatten_point(jc_rel)


def simple_seq_knee_hard(seq_len):
    """Generator which returns a sequence of seq_len of sequential timesteps. Data is in format [child_by_parent(
    source, 1), child_by_parent(source, 2), ...], hard_exo_joint_by_parent()"""
    for source in markers:
        for i in range(0, len(source.get_marker("knee")) - seq_len, seq_len):
            data = [child_by_parent(source, i + n, "thigh", "shank") for n in range(seq_len)]
            label = hard_exo_joint_by_parent(source, i + seq_len - 1, "thigh")
            yield data, label

