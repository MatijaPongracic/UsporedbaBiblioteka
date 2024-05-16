import os
import tensorflow as tf
import numpy as np
import cv2
import time

interpreter = tf.lite.Interpreter(model_path = "3.tflite")
interpreter.allocate_tensors()

folders = ["videos\\baseball", "videos\\baseball", "videos\\guitar",
           "videos\\jumpingJacks", "videos\\bowling", "videos\\jumpingRope"]
attempts = 0

for video in folders:
    folder_path = video
    frame_files = [f for f in os.listdir(folder_path)]

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    start = time.time()
    frames = 0

    for frame_file in frame_files:
        image = cv2.imread(os.path.join(folder_path, frame_file))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = tf.image.resize_with_pad(np.expand_dims(image_rgb, axis=0), 192, 192)
        input_image = tf.cast(img, dtype=tf.float32)

        interpreter.set_tensor(input_details[0]["index"], np.array(input_image))
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]["index"])

        frames += 1

    end = time.time()
    vrijeme = end - start
    fps = frames / vrijeme

    if attempts > 0:  #Prvo pokretanje se preskace zbog pocetnog sporijeg djelovanja
        print(f"Prosjecni FPS ({video}): {fps:.4f}")

    attempts += 1
