# Face and landmarks detector

import cv2
import mediapipe as mp
import numpy as np
import math

from scipy.spatial.transform import Rotation

lk_params = dict(winSize=(101, 101), maxLevel=15, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001))


def constrainPoint(p, w, h):
    p = (min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1))
    return p


class FaceDetector:
    def __init__(self):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(True, 1, True, 0.2, 0.5)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(200, 200, 0))
        self.cap = None
        self.stream_started = False

    def stabilizeVideoStream(self, frame, landmarks):
        img2Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not self.stream_started:
            self.points2Prev = np.array(landmarks, np.float32)
            self.img2GrayPrev = np.copy(img2Gray)
            self.stream_started = True

        points2Next, st, err = cv2.calcOpticalFlowPyrLK(self.img2GrayPrev, img2Gray, self.points2Prev,
                                                        np.array(landmarks, np.float32),
                                                        **lk_params)
        for k in range(0, len(landmarks)):
            d = cv2.norm(np.array(landmarks[k]) - points2Next[k])
            alpha = math.exp(-d * d / 50)
            landmarks[k] = (1 - alpha) * np.array(landmarks[k]) + alpha * points2Next[k]
            landmarks[k] = constrainPoint(landmarks[k], frame.shape[1], frame.shape[0])
            landmarks[k] = (int(landmarks[k][0]), int(landmarks[k][1]))

        self.points2Prev = np.array(landmarks, np.float32)
        self.img2GrayPrev = img2Gray

    # Read (target) image with alpha
    def load_target_img(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        return img

    def load_target_video(self, video_path):
        self.cap = cv2.VideoCapture(video_path)

    def get_target_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    # Draw landmarks in the image
    def drawLandmarks(self, img, landmarks):
        out = np.copy(img)

        self.mpDraw.draw_landmarks(out, landmarks, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)
        self.mpDraw.draw_landmarks(out, landmarks, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)

        return out

    def calculateRotation(self, lmks):
        indexPairs = []
        offsets = []
        faceMesh_result = lmks.squeeze()

        indexPairs += [(197, 4)]
        offsets += [[0, -1, 0]]
        zIndexPairs = [(356, 264), (127, 34)]
        indexPairs += zIndexPairs
        offsets += [[0, 0, 1], [0, 0, 1]]

        referencePoints = []
        comparePoints = []

        for (index1, index2), offset in zip(indexPairs, offsets):
            point1 = faceMesh_result[index1]
            point2 = faceMesh_result[index2]

            referencePoints.append(offset)
            comparePoints.append(point1 - point2)
        rotation, rmsd = Rotation.align_vectors(referencePoints, comparePoints)
        angles = Rotation.as_rotvec(rotation) * 180 / np.pi
        return angles

    # Find face landmarks with mediapipe

    def rotateLandmarks(self, lmks, angles):
        rotation = Rotation.from_rotvec(angles * np.pi / 180)
        rotated = rotation.apply(lmks)
        return rotated

    def scaleLandmarks(self, lmks, scale):
        height, width = scale

        faceMesh_result = lmks.squeeze()
        scaled = faceMesh_result * (width, height)
        scaled = scaled.astype(np.int32)
        return scaled

    def find_face_landmarks(self, img, draw=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        if not results.multi_face_landmarks:
            print('Face not detected!!!')
            return [], None, None

        allLandmarks = []
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            if draw:
                self.drawLandmarks(img, face_landmarks)

            values = np.array(face_landmarks.landmark)
            face_keypnts = np.zeros((len(values), 3))

            for idx, value in enumerate(values):
                face_keypnts[idx][0] = value.x
                face_keypnts[idx][1] = value.y
                face_keypnts[idx][2] = value.z
        return face_keypnts, img, face_landmarks
