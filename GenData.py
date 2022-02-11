import math
import numpy as np
from scipy.spatial.transform import Rotation as R


def randomOnSphere(dist=1, center_x=0, center_y=0, center_z=0):
    # Generates a random point on the surface of a 3D sphere centered at (center_x, center_y, center_z) with radius dist
    x = dist
    y = 0
    z = 0

    maxrot = math.pi * 2
    rot_x = np.random.uniform(0, maxrot)
    rot_y = np.random.uniform(0, maxrot)
    rot_z = np.random.uniform(0, maxrot)

    r = R.from_rotvec([rot_x, rot_y, rot_z])
    return r.apply([[x, y, z]])[0] + np.array([center_x, center_y, center_z])


def genSimpleSeq(seq_len):  # Generates a sequence of seq_len points randomly rotated around a random point
    TRA_RANGE = 10  # Maximum translation of the center point from the origin, per axis
    RAD_RANGE = 5  # Maximum radius of the sphere
    # Sequence will be scaled down by factor of 1/TRA+RAD, so to ensure all points are within -1 - 1 range

    center = np.array([np.random.uniform(0, TRA_RANGE) for n in range(3)])
    rad = np.random.uniform(0, RAD_RANGE)
    points = np.array([randomOnSphere(rad, center[0], center[1], center[2])/(TRA_RANGE+RAD_RANGE) for n in range(seq_len)])

    return points.flatten(), center/(TRA_RANGE+RAD_RANGE), rad/(TRA_RANGE+RAD_RANGE)


DAT_LEN = 20000
VAL_LEN = 5000

seq_len = 50

f = open("generated_simple_seq_flat_norm_len50.csv", "w")
for i in range(DAT_LEN):
    data, label, dist = genSimpleSeq(seq_len)
    f.write(",".join([str(n) for n in data]) + "," + ",".join([str(n) for n in label])+","+str(dist)+"\n")
f.close()
f = open("generated_simple_seq_flat_norm_len50-val.csv", "w")
for i in range(VAL_LEN):
    data, label, dist = genSimpleSeq(seq_len)
    f.write(",".join([str(n) for n in data]) + "," + ",".join([str(n) for n in label])+","+str(dist)+"\n")
f.close()



# print(randomOnSphere() + np.array([1, 1, 1]))
