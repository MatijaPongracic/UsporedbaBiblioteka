import os
import cv2
import mediapipe as mp
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode = True, min_detection_confidence = 0.5)

folders = ["baseball", "baseball", "guitar", "jumpingJacks", "bowling", "jumpingRope"]
attempts = 0

for video in folders:
    folder_path = video
    frame_files = [f for f in os.listdir(folder_path)]

    start = time.time()
    frames = 0

    for frame_file in frame_files:
        image = cv2.imread(os.path.join(folder_path, frame_file))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        frames += 1

    end = time.time()
    vrijeme = end - start
    fps = frames / vrijeme

    if attempts > 0:  #Prvo pokretanje se preskace zbog pocetnog sporijeg djelovanja
        print(f"Prosjecni FPS ({video}): {fps:.4f}")

    attempts += 1

pose.close()