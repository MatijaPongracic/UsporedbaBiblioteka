import json
import cv2
import os
import scipy.io
from math import sqrt

def distance(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

tolerance = 0.1
videos = {"videos\\baseball":"matdata\\0005.mat",
          "videos\\jumpingJacks":"matdata\\1083.mat",
          "videos\\guitar":"matdata\\1951.mat",
          "videos\\bowling":"matdata\\0532.mat",
          "videos\\jumpingRope":"matdata\\0989.mat"}

json_files = {"videos\\baseball":"OP_baseball",
              "videos\\jumpingJacks":"OP_jumpingJacks",
              "videos\\guitar":"OP_guitar",
              "videos\\bowling":"OP_bowling",
              "videos\\jumpingRope":"OP_jumpingRope"}

for key, value in videos.items():
    mat_file_path = value
    mat_data = scipy.io.loadmat(mat_file_path)

    action = mat_data["action"]
    x = mat_data["x"]
    y = mat_data["y"]
    visibility = mat_data["visibility"]
    train = mat_data["train"]
    bbox = mat_data["bbox"]
    dimensions = mat_data["dimensions"]
    nframes = mat_data["nframes"]

    joint_match = {1:2, #desno rame
                   2:5, #lijevo rame
                   3:3, #desni lakat
                   4:6, #lijevi lakat
                   5:4, #desni zglob
                   6:7, #lijevi zglob
                   7:9, #desni kuk
                   8:12, #lijevi kuk
                   9:10, #desno koljeno
                   10:13, #lijevo koljeno
                   11:11, #desni glezanj
                   12:14} #lijevi glezanj

    frame_files = [f for f in os.listdir(key)]
    frame_points = [p for p in os.listdir(os.path.join("OP_data1", json_files.get(key)))]

    low = 0
    mid_low = 0
    mid_high = 0
    high = 0
    for i in range(len(frame_files)):
        image = cv2.imread(os.path.join(key, frame_files[i]))
        with open(os.path.join("OP_data1", json_files.get(key), frame_points[i])) as f:
            data = json.load(f)

        tocno = 0
        visible_keypoints = 0
        if visibility[i, 8] == 0:
            dt = distance(x[i, 1], y[i, 1], x[i, 2], y[i, 2])  # udaljenost lijevog i desnog ramena
        else:
            dt = distance(x[i, 1], y[i, 1], x[i, 8], y[i, 8])  # udaljenost desnog ramena i lijevog kuka

        for key1, value1 in joint_match.items():
            if visibility[i, key1] == 0:
                continue  # Ako je tocka nevidljiva, ne racuna se
            else:
                visible_keypoints += 1
                try:
                    a = data['people'][0]['pose_keypoints_2d'][value1 * 3]
                    b = data['people'][0]['pose_keypoints_2d'][value1 * 3 + 1]
                    c = data['people'][0]['pose_keypoints_2d'][value1 * 3 + 2]
                except:  # ukoliko nijedan zglob nije prepoznat
                    a = 0.0
                    b = 0.0
                if c < 0.1:
                    a = 0.0
                    b = 0.0

                d = distance(x[i, key1], y[i, key1], a, b)
                if d < tolerance * dt:
                    tocno += 1

                #cv2.circle(image, (int(a), int(b)), 5, (0, 0, 255), -1)
                #cv2.circle(image, (int(x[i, key1]), int(y[i, key1])), 5, (0, 255, 0), -1)
        PDJ = float(tocno / visible_keypoints)

        if PDJ < 0.25:
            low += 1
        elif PDJ < 0.5:
            mid_low += 1
        elif PDJ < 0.75:
            mid_high += 1
        else:
            high += 1

        #cv2.imshow("Image", image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    print(key + ":")
    print(f"0%-25%: {low}")
    print(f"25%-50%: {mid_low}")
    print(f"50%-75%: {mid_high}")
    print(f"75%-100%: {high}\n")
