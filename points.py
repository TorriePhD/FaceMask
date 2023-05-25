import numpy as np
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

oval = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,140,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
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
