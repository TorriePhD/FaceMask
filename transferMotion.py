from Detector import FaceDetector
from MaskGenerator import MaskGenerator
import numpy as np

detector = FaceDetector()
maskGenerator = MaskGenerator()

def transferMotion(srcFrame,dstFrame):
    target_landmarks, _, target_face_landmarks = detector.find_face_landmarks(
        dstFrame)
    if len(target_landmarks) == 0:
        return
    target_rotation = detector.calculateRotation(target_landmarks)
    target_landmarks = detector.scaleLandmarks(np.array(target_landmarks)[:, :2], dstFrame.shape[:-1])
    maskGenerator.calculateTargetInfo(dstFrame, target_landmarks)
    orglandmarks, image, _ = detector.find_face_landmarks(srcFrame)
    if len(orglandmarks) == 0:
        return
    rotation = detector.calculateRotation(orglandmarks)
    landmarks = detector.rotateLandmarks(orglandmarks, rotation)
    landmarks = detector.rotateLandmarks(landmarks, -target_rotation)
    landmarks = detector.scaleLandmarks(landmarks[:, :2], srcFrame.shape[:-1])
    allLandmarks = detector.rotateLandmarks(orglandmarks, rotation)
    allLandmarks = detector.rotateLandmarks(allLandmarks, -target_rotation)
    allLandmarks = detector.scaleLandmarks(allLandmarks[:, :2], srcFrame.shape[:-1])
    orglandmarks = detector.scaleLandmarks(orglandmarks[:, :2], srcFrame.shape[:-1])
    maskGenerator.applyTargetMask(srcFrame, np.zeros((srcFrame.shape[0] * 2, srcFrame.shape[1] * 2,3)), landmarks, allLandmarks, orglandmarks)
    output = maskGenerator.applyTargetMaskToTarget(landmarks)
    _, _, outLmks1 = detector.find_face_landmarks(output)
    display = detector.drawLandmarks(np.zeros_like(output), outLmks1)
    return output,display