# Emotion recognition
import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

emotion_labels ={0:'angry', 1:'disgust', 2:'fear', 3:'happy',
                 4:'sad', 5:'surprise', 6:'neutral'}

emotion_model_path = "./models/emotion/fer2013_mini_XCEPTION.102-0.66.hdf5"
frame_window = 10
emotion_offsets = (20, 40)

dir_name, _ = os.path.split(os.path.abspath(__file__))
haar_cascade_path = dir_name + "/haarcascade_frontalface_default.xml"


def preprocess_gray_faces(x, gray=True):
    x = x.astype("float32")
    x = x / 255.0
    if gray:
        x = x - 0.5
        x = x * 2.0
    return x


def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off, = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)


def draw_text(coordinates, image_array, text, color, x_offset = 0, y_offset=0,
              font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv.putText(image_array, text, (x+x_offset, y+y_offset),cv.FONT_HERSHEY_SIMPLEX,
               font_scale, color, thickness, cv.LINE_AA)


def get_colors(num_classes):
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes).tolist())
    colors = np.asarray(colors) * 255
    return colors

