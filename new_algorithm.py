import pyrealsense2 as rs
import numpy as np
import cv2
import time
import mediapipe as mp
import tkinter as tk
from tkinter import font
from PIL import Image, ImageTk

def set_speed(x):
    global robot_speed, count, prev_x, speed_var, Nula
    if Nula:
        if robot_speed == 0.0:
            Nula = False
        else:
            robot_speed = round(robot_speed - 0.1, 1)
            speed_var.set(f"{int(robot_speed * 100)}%\n{x}")
        return
    if robot_speed == 0.0 and x < 0.8: # vraćanje u zonu 2
        count = 0
        return
    if x == robot_speed:
        count = 0
        # Nula = False
        return
    if prev_x == x:
        count += 1
        if count >= 2: # 2 kadra zaredom moraju dati istu brzinu da ju robot prihvati
            if x == 0.0:
                Nula = True
                robot_speed = round(robot_speed - 0.1, 1)
            elif x < robot_speed:
                robot_speed = round(robot_speed - 0.1, 1)
            else:
                robot_speed = round(robot_speed + 0.1, 1)
            speed_var.set(f"{int(robot_speed * 100)}%\n{x}")
            count = 0
    else:
        count = 1
    prev_x = x

def update_frame():
    global prev_time, prev_y, pipeline_started, pose

    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        root.after(10, update_frame)
        return

    color_image = np.asanyarray(color_frame.get_data())
    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb_image)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(color_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        right_foot_landmark = results.pose_landmarks.landmark[31] # right heel 29
        left_foot_landmark = results.pose_landmarks.landmark[32] # left heel 30

        # desna y-koordinata
        if right_foot_landmark.visibility < 0.5:
            right_y = None
        elif right_foot_landmark.y > 1 or right_foot_landmark.y < 0:
            right_y = None
        else:
            right_y = right_foot_landmark.y

        # lijeva y-koordinata
        if left_foot_landmark.visibility < 0.5:
            left_y = None
        elif left_foot_landmark.y > 1 or left_foot_landmark.y < 0:
            left_y = None
        else:
            left_y = left_foot_landmark.y

        # srednja y-koordinata
        try:
            y = (right_y + left_y) / 2
        except TypeError:
            y = None
        try:
            dy = y - prev_y
        except TypeError:
            dy = None
        prev_y = y

        # brzina točke na slici
        try:
            w = dy / (current_time - prev_time)
        except TypeError:
            w = None

        # stvarna udaljenost
        try:
            x = r0 / (2 * y - 1)
        except TypeError:
            x = None

        # stvarna brzina
        try:
            v = (-2 * r0 * w) / (2 * y - 1) ** 2
        except:
            v = None

        # algoritam
        if x is None:
            if robot_speed >= 0.6: # covjek izlazi iz vidnog polja
                set_speed(1.0)
            else: # zona 0
                set_speed(0.0)

        elif x >= r2: # okolina
            set_speed(1.0)

        elif x < r2 and x >= r1: # zona 2
            if v is not None:
                if -v <= vm:
                    set_speed(0.8)
                elif -v >= (vt + vbh) / 2:
                    set_speed(0.0)
                elif -v >= (vbh + vh) / 2:
                    set_speed(0.3)
                else:
                    set_speed(0.6)

        else: # zona 1
            if v is not None:
                if -v <= vm:
                    set_speed(0.3)
                elif -v >= (vbh + vh) / 2:
                    set_speed(0.0)
                else:
                    set_speed(0.1)
    else: # nema covjeka
        if robot_speed != 0.0:
            set_speed(1.0)

    prev_time = current_time

    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    color_image = cv2.resize(color_image, (960, 720))
    img = Image.fromarray(color_image)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)

res_h = 640
res_v = 480

rk = 1
r0 = 2
r1 = 3.5
r2 = 5.5

vm = 0.1
vh = 1.2
vbh = 1.7
vt = 2.5
v_max = 1.5

global robot_speed, count, prev_x, speed_var, Nula
robot_speed = 1.0
count = 0
prev_x = 1.0
Nula = False

printw = False
pipeline_started = False
pose = None

try:
    while True:
        try:
            # Configure color stream only
            pipeline = rs.pipeline()
            config = rs.config()

            # Get device product line for setting a supporting resolution
            pipeline_wrapper = rs.pipeline_wrapper(pipeline)
            pipeline_profile = config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()
            device_product_line = str(device.get_info(rs.camera_info.product_line))
            break

        except RuntimeError:
            if not printw:
                print("Čekam spajanje s kamerom...")
            printw = True

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == "RGB Camera":
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.color, res_h, res_v, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    pipeline_started = True

    prev_time = time.time()
    prev_y = None

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    # Tkinter setup
    root = tk.Tk()
    root.title("RealSense Viewer")
    root.geometry("1440x810")

    # Create left frame for video
    left_frame = tk.Frame(root, width=960, height=720)
    left_frame.grid(row=0, column=0, padx=10, pady=10)

    video_label = tk.Label(left_frame)
    video_label.pack()

    # Create right frame for speed display
    right_frame = tk.Frame(root, width=480, height=720)
    right_frame.grid(row=0, column=1, padx=10, pady=10)

    speed_var = tk.StringVar()
    speed_var.set("100%")

    speed_font = font.Font(size=80)
    speed_label = tk.Label(right_frame, textvariable=speed_var, font=speed_font)
    speed_label.pack()

    root.after(10, update_frame)
    root.mainloop()

except KeyboardInterrupt:
    print("Program prekinut")

finally:
    if pipeline_started:
        pipeline.stop()
    if pose:
        pose.close()
    cv2.destroyAllWindows()
