import cv2
import numpy as np
from Detector import FaceDetector
from MaskGenerator import MaskGenerator
from pathlib import Path

detector = FaceDetector()
maskGenerator = MaskGenerator()

def showImages(actual, target, output1, output2):
    img_actual = np.copy(actual)
    img_target = np.copy(target)
    img_out1 = np.copy(output1)
    img_out2 = np.copy(output2)
    frame = np.zeros((max(img_actual.shape[0], img_target.shape[0],img_out1.shape[0],img_out2.shape[0]), img_actual.shape[1] + img_target.shape[1] + img_out1.shape[1] + img_out2.shape[1], 3), dtype=np.uint8)

    frame[:img_actual.shape[0], :img_actual.shape[1]] = img_actual
    frame[:img_target.shape[0], img_actual.shape[1]:img_actual.shape[1] + img_target.shape[1]] = img_target
    frame[:img_out1.shape[0], img_actual.shape[1] + img_target.shape[1]:img_actual.shape[1] + img_target.shape[1] + img_out1.shape[1]] = img_out1
    frame[:img_out2.shape[0], img_actual.shape[1] + img_target.shape[1] + img_out1.shape[1]:img_actual.shape[1] + img_target.shape[1] + img_out1.shape[1] + img_out2.shape[1]] = img_out2

    frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
    cv2.imshow('Face Mask', frame)

# Target
#target_image, target_alpha = detector.load_target_img("images/cage.png")
# target_image = detector.load_target_img("images/drew.jpg")
drive = Path("C:/Users/Shad Torrie/Documents/compute/DoorData")
np.random.seed(0)
for i in range(0,100):
    # np.random.seed(1)
    randomVideo = np.random.choice(list(drive.rglob('*.mp4')))
    target_image = detector.load_target_video(str(randomVideo))


    cap = cv2.VideoCapture(str(np.random.choice(list(drive.rglob('*.mp4')))))
    print("starting New video")
    # cap = cv2.VideoCapture(0)
    # target_image = detector.load_target_video("images/casey.mp4")
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
        if not success:
            break
        # frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        orglandmarks, image, face_landmarks,allLandmarks,Unscaledlmks = detector.find_face_landmarks(frame)
        if len(orglandmarks) == 0:
            continue
        rotation = detector.calculateRotation(Unscaledlmks)
        landmarks = detector.rotateLandmarks(orglandmarks, rotation)
        landmarks = detector.rotateLandmarks(landmarks, -target_rotation)
        landmarks = detector.scaleLandmarks(landmarks[:,:2],frame.shape[:-1])
        # detector.stabilizeVideoStream(frame, landmarks)
        allLandmarks = detector.rotateLandmarks(allLandmarks, rotation)
        allLandmarks = detector.rotateLandmarks(allLandmarks, -target_rotation)
        allLandmarks = detector.scaleLandmarks(allLandmarks[:, :2], frame.shape[:-1])
        orglandmarks = detector.scaleLandmarks(orglandmarks[:, :2], frame.shape[:-1])
        output = maskGenerator.applyTargetMask(frame,np.zeros_like(frame), landmarks,allLandmarks,orglandmarks)
        output2 = maskGenerator.applyTargetMaskToTarget(landmarks)
        dstLandmarks,_,face_landmarksNew,_,_ = detector.find_face_landmarks(output2)
        if len(dstLandmarks) == 0:
            continue
        dstLandmarks = detector.scaleLandmarks(dstLandmarks[:, :2], output2.shape[:-1])
        image_out = detector.drawLandmarks(np.zeros_like(output2), face_landmarksNew)
        image_out1 = detector.drawLandmarks(np.zeros_like(output), face_landmarks)
        showImages(image, image_out1,image_out , output2)

        cv2.waitKey(1)