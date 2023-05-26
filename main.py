import cv2
import numpy as np
from Detector import FaceDetector
from MaskGenerator import MaskGenerator
from pathlib import Path
from transferMotion import transferMotion
import imageio

detector = FaceDetector()
maskGenerator = MaskGenerator()
VIDEO = False
DOOR = False
def showImages(*images):
    max_height = max(image.shape[0] for image in images)
    total_width = sum(image.shape[1] for image in images)
    frame = np.zeros((max_height, total_width, 3), dtype=np.uint8)

    current_width = 0
    for image in images:
        frame[:image.shape[0], current_width:current_width+image.shape[1]] = image
        current_width += image.shape[1]

    frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
    cv2.imshow('Concatenated Images', frame)
    return frame

# Target
#target_image, target_alpha = detector.load_target_img("images/cage.png")
# target_image = detector.load_target_img("images/drew.jpg")
doorData = Path("C:/Users/Shad Torrie/Documents/compute/DoorData")
drive = Path("E:/Authentiface Dataset/")
# np.random.seed(3)
for i in range(0,100):
    # np.random.seed(1)
    if DOOR:
        randomDestination = np.random.choice(list(doorData.rglob('*.mp4')))
    else:
        randomDestination = np.random.choice(list(drive.rglob('*.mkv')))
    detector.load_target_video(str(randomDestination))
    if DOOR:
        randomSource = np.random.choice(list(doorData.rglob('*.mp4')))
    else:
        randomSource = np.random.choice(list(drive.rglob('*.mkv')))
    # randomSource = np.random.choice(list(Path("C:/Users/Shad Torrie/Documents/compute/DoorData/Truman/eyebrowLift/").rglob('*.mp4')))
    # randomSource = Path("C:/Users/Shad Torrie/Documents/compute/DoorData/Truman/eyebrowLift/2021-11-30T15_32_41.113600.mp4")

    if "true_fails" in str(randomSource):
        continue
    cap = cv2.VideoCapture(str(randomSource))
    # videoWriter = None
    frames = []
    success, target_image = detector.get_target_frame()
    if not success:
        continue
    # print(f"starting {randomSource}, {randomDestination}")
    while True:
        success, frame = cap.read()
        if not success:
            break
        if VIDEO:
            success, target_image = detector.get_target_frame()
            if not success:
                break
        output,displayLmks = transferMotion(frame,target_image)
        frame = showImages(frame, target_image,output,displayLmks)
        # if videoWriter is None:
        #     videoWriter = cv2.VideoWriter(f'videos/{randomSource.parent.parent.name}_{randomSource.parent.name}_to_{randomDestination.parent.parent.name}_{randomDestination.parent.name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (frame.shape[1], frame.shape[0]))
        # videoWriter.write(frame)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(1) & 0xFF == ord('n'):
            break
    # videoWriter.release()
    # imageio.mimsave(f'gifs/{randomSource.parent.parent.name}_{randomSource.parent.name}_to_{randomDestination.parent.parent.name}_{randomDestination.parent.name}.gif', frames, fps=15)
    # print(f'gifs/{randomSource.parent.parent.name}_{randomSource.parent.name}_to_{randomDestination.parent.parent.name}_{randomDestination.parent.name}.gif')