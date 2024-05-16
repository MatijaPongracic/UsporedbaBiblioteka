from ultralytics import YOLO
import os
import cv2
import time

model = YOLO("yolov8n-pose.pt")

folders = ["videos\\baseball", "videos\\baseball", "videos\\guitar",
           "videos\\jumpingJacks", "videos\\bowling", "videos\\jumpingRope"]
attempts = 0

for video in folders:
    folder_path = video
    frame_files = [f for f in os.listdir(folder_path)]

    start = time.time()
    frames = 0

    for frame_file in frame_files:
        image = cv2.imread(os.path.join(folder_path, frame_file))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(source=image_rgb, show=False, conf=0.3, save=False)

        frames += 1

    end = time.time()
    vrijeme = end - start
    fps = frames / vrijeme

    if attempts > 0:  #Prvo pokretanje se preskace zbog pocetnog sporijeg djelovanja
        print(f"Prosjecni FPS ({video}): {fps:.4f}")

    attempts += 1
