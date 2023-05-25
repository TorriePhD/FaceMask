import cv2
import numpy as np
from Detector import FaceDetector
from MaskGenerator import MaskGenerator
from pathlib import Path

cap = cv2.VideoCapture(0)
detector = FaceDetector()
maskGenerator = MaskGenerator()

def showImages(actual, target, output1, output2):
    img_actual = np.copy(actual)
    img_target = np.copy(target)
    img_out1 = np.copy(output1)
    img_out2 = np.copy(output2)
    # 640x480 -> 360x480
    img_actual = img_actual[:, 140:500]
    img_out1 = img_out1[:, 140:500]
    # 480x640 -> 360x480
    img_target = cv2.resize(img_target, (360, 480), interpolation=cv2.INTER_AREA)
    img_out2 = cv2.resize(img_out2, (360, 480), interpolation=cv2.INTER_AREA)

    h1 = np.concatenate((img_actual, img_target, img_out1, img_out2), axis=1)

    cv2.imshow('Face Mask', h1)

# Target
#target_image, target_alpha = detector.load_target_img("images/cage.png")
# target_image = detector.load_target_img("images/drew.jpg")
drive = Path("E:\Authentiface Dataset")
randomVideo = np.random.choice(list(drive.rglob('*.mkv')))
target_image = detector.load_target_video(str(randomVideo))
#target_image, target_alpha = detector.load_target_img("images/trump.png")
#target_image, target_alpha = detector.load_target_img("images/kim.png")
#target_image, target_alpha = detector.load_target_img("images/putin.png")
target_landmarks, _, target_face_landmarks,targetallLandmarks,targetUnscaledlmks = detector.find_face_landmarks(target_image)
target_rotation = detector.calculateRotation(targetUnscaledlmks)
target_image_out = detector.drawLandmarks(target_image, target_face_landmarks)
target_landmarks = detector.scaleLandmarks(np.array(target_landmarks)[:,:2],target_image.shape[:-1])
maskGenerator.calculateTargetInfo(target_image, target_landmarks)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    landmarks, image, face_landmarks,allLandmarks,Unscaledlmks = detector.find_face_landmarks(frame)
    if len(landmarks) == 0:
        continue
    rotation = detector.calculateRotation(Unscaledlmks)
    landmarks = detector.rotateLandmarks(landmarks, rotation)
    landmarks = detector.rotateLandmarks(landmarks, -target_rotation)
    landmarks = detector.scaleLandmarks(landmarks[:,:2],frame.shape[:-1])
    detector.stabilizeVideoStream(frame, landmarks)
    allLandmarks = detector.rotateLandmarks(allLandmarks, rotation)
    allLandmarks = detector.rotateLandmarks(allLandmarks, -target_rotation)
    allLandmarks = detector.scaleLandmarks(allLandmarks[:, :2], frame.shape[:-1])
    output = maskGenerator.applyTargetMask(np.zeros_like(frame), landmarks,allLandmarks)
    output2 = maskGenerator.applyTargetMaskToTarget(landmarks)
    _,_,face_landmarksNew,_,_ = detector.find_face_landmarks(output2)
    image_out = detector.drawLandmarks(np.zeros_like(output2), face_landmarksNew)
    image_out1 = detector.drawLandmarks(np.zeros_like(output), face_landmarks)
    showImages(image, image_out,image_out1 , output2)

    cv2.waitKey(1)