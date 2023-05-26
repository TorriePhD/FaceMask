# Face Mask generator

import cv2
import numpy as np
import mediapipe as mp
from points import oval,allPoints,triangles,mouthTrianges

class MaskGenerator:
    def __init__(self):
        self.target = {}


    # Check if a point is inside a rectangle
    def rectContains(self, rect, point):
        return point[0] >= rect[0] and point[1] >= rect[1] and point[0] <= rect[2] and point[1] <= rect[3]

    # Apply affine transform calculated using srcTri and dstTri to src
    def applyAffineTransform(self, src, srcTri, dstTri, size):

        warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
        dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None,
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return dst

    # Warps triangular regions from img1 and img2 to img
    def warpTriangle(self, img1, img2, t1, t2):
        # Find bounding rectangle for each triangle
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))
        # Offset points by left top corner of the respective rectangles
        t1Rect = []
        t2Rect = []
        t2RectInt = []

        for i in range(0, 3):
            t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
            t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
            t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)
        img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

        size = (r2[2], r2[3])
        #if any dimension is 0, return
        if img1Rect.shape[0] ==0 or img1Rect.shape[1] == 0 or img1Rect.shape[2] == 0:
            return
        img2Rect = self.applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
        img2Rect = img2Rect * mask
        if img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]].shape != mask.shape:
            return
        img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
                (1.0, 1.0, 1.0) - mask)
        img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect

    def calculateTargetInfo(self, target_image, target_landmarks):


        sizeImg1 = target_image.shape
        self.target["image"] = target_image
        self.target["width"] = sizeImg1[1]
        self.target["height"] = sizeImg1[0]
        self.target["landmarks"] = target_landmarks

    def applyTargetMask(self,frame, actual_img, actual_landmarks,allLandmarks,orgLandmarks):
        warped_img = np.copy(actual_img)
        ovalPoints =  np.array(allLandmarks)[oval].reshape((-1, 1, 2))
        mask1 = np.zeros((warped_img.shape[0], warped_img.shape[1]), dtype=np.float32)
        mask1 = cv2.merge((mask1, mask1, mask1))

        cv2.fillPoly(mask1, [ovalPoints], (255,255,255))
        # Warp the triangles
        for i in range(0, len(triangles)):
            t1 = []
            t2 = []
            for j in range(0, 3):
                t1.append(self.target["landmarks"][triangles[i][j]])
                t2.append(actual_landmarks[triangles[i][j]])
            self.warpTriangle(self.target["image"], warped_img, t1, t2)
        for i in range(0, len(mouthTrianges)):
            t1 = []
            t2 = []
            for j in range(0, 3):
                t1.append(orgLandmarks[mouthTrianges[i][j]])
                t2.append(actual_landmarks[mouthTrianges[i][j]])

            self.warpTriangle(frame, warped_img, t1, t2)
        # Alpha blending of the two images
        temp1 = np.multiply(warped_img, (mask1 * (1.0 / 255)))



        self.temp1 = temp1
        self.mask1 = mask1
    def applyTargetMaskToTarget(self, actual_landmarks):
        t_w, t_h = (self.target["width"], self.target["height"])
        # 0. Calculate homography actual_landmarks -> target_landmarks
        pts_src = np.array([[p[0], p[1]] for p in actual_landmarks])
        dst_src = np.array([[p[0], p[1]] for p in self.target["landmarks"]])
        h, _ = cv2.findHomography(pts_src, dst_src)

        # 1. Apply homography to actual_img
        im_out_temp1 = cv2.warpPerspective(self.temp1, h, (t_w, t_h))
        im_out_mask1 = cv2.warpPerspective(self.mask1, h, (t_w, t_h))

        # 2. Overlap result in target_image
        mask2 = (255.0, 255.0, 255.0) - im_out_mask1

        # 3. Apply homography in the opposite direction
        target_image = np.copy(self.target["image"])

        # 4. Alpha blending of the two images
        temp2 = np.multiply(target_image, (mask2*1/255))
        output = im_out_temp1 + temp2
        return np.uint8(output)
