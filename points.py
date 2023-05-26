import numpy as np
import mediapipe as mp
from tqdm import tqdm
mp_face_mesh = mp.solutions.face_mesh

oval = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
FACEMESH_NOSE = [(168, 6), (6, 197), (197, 195), (195, 5),
                           (5, 4), (4, 1), (1, 19), (19, 94), (94, 2), (98, 97),
                           (97, 2), (2, 326), (326, 327), (327, 294),
                           (294, 278), (278, 344), (344, 440), (440, 275),
                           (275, 4), (4, 45), (45, 220), (220, 115), (115, 48),
                           (48, 64), (64, 98)]
# FACEMESH_LIPS = list(set(np.array(list(mp_face_mesh.FACEMESH_LIPS)).flatten()))
FACEMESH_LIPS = [[48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59],[60], [61], [62], [63], [64], [65], [66], [67]]
FACEMESH_LIPS = list(np.array(FACEMESH_LIPS).squeeze())

FACEMESH_LEFT_EYE = list(set(np.array(list(mp_face_mesh.FACEMESH_LEFT_EYE)).flatten()))

FACEMESH_LEFT_IRIS = list(set(np.array(list(mp_face_mesh.FACEMESH_LEFT_IRIS)).flatten()))
FACEMESH_LEFT_EYEBROW = list(set(np.array(list(mp_face_mesh.FACEMESH_LEFT_EYEBROW)).flatten()))
FACEMESH_RIGHT_EYE = list(set(np.array(list(mp_face_mesh.FACEMESH_RIGHT_EYE)).flatten()))
FACEMESH_RIGHT_EYEBROW = list(set(np.array(list(mp_face_mesh.FACEMESH_RIGHT_EYEBROW)).flatten()))
FACEMESH_RIGHT_IRIS = list(set(np.array(list(mp_face_mesh.FACEMESH_RIGHT_IRIS)).flatten()))
FACEMESH_FACE_OVAL = list(set(np.array(list(mp_face_mesh.FACEMESH_FACE_OVAL)).flatten()))
FACEMESH_NOSE = list(set(np.array(FACEMESH_NOSE).flatten()))

allPoints = FACEMESH_LIPS + FACEMESH_LEFT_EYE  + FACEMESH_LEFT_EYEBROW + FACEMESH_RIGHT_EYE + FACEMESH_RIGHT_EYEBROW + FACEMESH_FACE_OVAL +FACEMESH_NOSE


FACEMESH_TESSELATION = np.array(list(mp_face_mesh.FACEMESH_TESSELATION))
print(FACEMESH_TESSELATION.shape)


def find_triangles(edges):
    connections = {}
    triangles = []

    # Build connections dictionary
    for edge in edges:
        node1, node2 = edge
        if node1 in connections:
            connections[node1].add(node2)
        else:
            connections[node1] = {node2}
        if node2 in connections:
            connections[node2].add(node1)
        else:
            connections[node2] = {node1}

    # Find triangles
    for edge1 in edges:
        node1, node2 = edge1
        if node2 in connections:
            for node3 in connections[node2]:
                if node3 in connections[node1]:
                    triangle = [node1, node2, node3]
                    triangle.sort()
                    if triangle not in triangles:
                        triangles.append(triangle)

    return triangles
triangles = np.array(find_triangles(FACEMESH_TESSELATION))
left_eye_triangles = [[33,7,246],[7,246,163],[246,163,161],[163,161,144],[161,160,144],[144,160,145],[160,159,145],[159,145,153],[159,158,153],[158,153,154],[158,157,154],[157,154,155],[157,173,155],[173,155,133]]
right_eye_triangles= [[362,398,382],[398,382,384],[382,381,384],[381,384,385],[381,385,380],[385,380,386],[380,386,374],[386,374,387],[374,387,373],[387,388,373],[388,373,390],[388,390,466],[466,249,263]]

mouthTrianges = [[78,95,191],[191,95,80],[95,88,80],[80,88,81],[88,81,178],[81,178,82],[178,82,87],[82,87,13],[87,13,14],[13,14,317],[312,13,317],[312,317,402],[312,311,402],[311,402,318],[311,318,310],[310,318,324],[415,310,324],[415,324,308]]
triangles = np.concatenate([triangles,left_eye_triangles,right_eye_triangles],axis=0)