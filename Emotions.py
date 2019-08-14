from statistics import mode

import cv2
import numpy as np
import tensorflow as tf

from libs.Emotions import emotion_recognition
from utils.inference import apply_offsets
from utils.inference import detect_faces
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input


# from libs.MTCNN import detect_face
class FacialExpressionRecognition:
    def __init__(self, img=None, use_haar_cascade=False):
        self.emotion_window = []

        self.use_haar_cascade = use_haar_cascade
        if self.use_haar_cascade:
            self.face_detection_model = load_detection_model(emotion_recognition.haar_cascade_path)
            # print('[INFO] Haar-cascade loaded')

        # else:
        #     # Use MTCNN face detector
        #     self.min_face_size = 50
        #     self.threshold = [0.6, 0.7, 0.7]
        #     self.scale_factor = 0.709
        #     with tf.Session().as_default() as sess:
        #         self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess=sess, model_path=None)

        if img is not None:
            self.img = img

        self.emotion_labels = emotion_recognition.emotion_labels
        global emotion_classifier
        emotion_classifier = tf.keras.models.load_model(emotion_recognition.emotion_model_path, compile=False)
        print('[Emotions] FER loaded')

    # def live_camera(self):
    #
    #     # cv2.namedWindow('Live Preview', cv2.WINDOW_AUTOSIZE)
    #     # cap = cv2.VideoCapture(0)
    #     # while cap.isOpened():
    #     # ret, frame = cap.read()
    #     frame = cv2.imread(self.img)
    #     gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #
    #     total_boxes, _ = detect_face.detect_face(frame, minsize = self.min_face_size,
    #                                                 pnet=self.pnet, rnet=self.rnet, onet=self.onet,
    #                                                 threshold=self.threshold, factor=self.scale_factor)
    #     emotion_window = []
    #     bounding_boxes = total_boxes[:, 0:4]
    #     nof_faces = total_boxes.shape[0]
    #
    #
    #     for i in range(nof_faces):
    #
    #         # cv2.rectangle(frame, int(x1), int(y1), int(x1+w), int(y1+h), (0,255,0), 2)
    #         x1 = int(bounding_boxes[i][0])
    #         y1 = int(bounding_boxes[i][1])
    #         x2 = int(bounding_boxes[i][2])
    #         y2 = int(bounding_boxes[i][3])
    #
    #         h = y2 - y1
    #         w = x2 - x1
    #
    #         print('x1', x1)
    #         print('x2', x2)
    #         print('y1', y1)
    #         print('y2', y2)
    #         print('h', h)
    #         print('w', w)
    #         x_off, y_off = 24, 90
    #
    #         _x1 = x1 - 24
    #         _x2 = x2 + 70
    #         _y1 = y1 - 55
    #         _y2 = y2 + 33
    #
    #         print('x1', _x1)
    #         print('x2',_x2)
    #         print('y1',_y1)
    #         print('y2',_y2)
    #
    #
    #         gray_face = gray_image[_y1:_y2, _x1:_x2]
    #         # print(gray_face.shape)
    #         cv2.imshow('gray', gray_face)
    #         try:
    #             gray_face = cv2.resize(gray_face, (64, 64), interpolation=cv2.INTER_CUBIC)
    #         except:
    #             continue
    #
    #         gray_face = emotion_recognition.preprocess_gray_faces(gray_face, True)
    #         gray_face = np.expand_dims(gray_face, 0)
    #         gray_face = np.expand_dims(gray_face, -1)
    #
    #         emotion_prediction = self.emotion_classifier.predict(gray_face)
    #         # print(emotion_prediction)
    #         emotion_probability = np.max(emotion_prediction)
    #         emotion_label_arg = np.argmax(emotion_probability)
    #         emotion_text = self.emotion_labels[emotion_label_arg]
    #         emotion_window.append(emotion_text)
    #
    #         if len(emotion_window) > 10:
    #             emotion_window.pop(0)
    #         try:
    #             emotion_mode = mode(emotion_window)
    #         except:
    #             continue
    #
    #         if emotion_text == 'angry':
    #             color = emotion_probability * np.asarray((255, 0, 0))
    #         elif emotion_text == 'sad':
    #             color = emotion_probability * np.asarray((0, 0, 255))
    #         elif emotion_text == 'happy':
    #             color = emotion_probability * np.asarray((255, 255, 0))
    #         elif emotion_text == 'surprise':
    #             color = emotion_probability * np.asarray((0, 255, 255))
    #         else:
    #             color = emotion_probability * np.asarray((0, 255, 0))
    #
    #         color = color.astype(int)
    #         color = color.tolist()
    #
    #         draw_bounding_box((x1, y1, w, h), frame, color)
    #         draw_text((x1, y1, w, h), frame, emotion_mode, color, 0, -45, 1, 1)
    #         cv2.imshow('Live Preview', frame)
    #         k = cv2.waitKey(0) & 0xFF
    #         if k == ord('q'):
    #             # cap.release()
    #             cv2.destroyAllWindows()
    #
    # def _live_camera(self):
    #     cv2.namedWindow('Live Camera', cv2.WINDOW_AUTOSIZE)
    #     video_capture = cv2.VideoCapture(0)
    #     emotion_window = []
    #
    #     while True:
    #         bgr_image = video_capture.read()[1]
    #         gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    #         rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    #         faces = detect_faces(self.face_detection_model, gray_image)
    #
    #         for face_coordinates in faces:
    #             x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_recognition.emotion_offsets)
    #             gray_face = gray_image[y1:y2, x1:x2]
    #
    #             try:
    #                 gray_face = cv2.resize(gray_face, (64, 64))
    #             except:
    #                 continue
    #
    #             gray_face = preprocess_input(gray_face, True)
    #             gray_face = np.expand_dims(gray_face, 0)
    #             gray_face = np.expand_dims(gray_face, -1)
    #
    #             emotion_prediction = emotion_classifier.predict(gray_face)
    #             emotion_probability = np.max(emotion_prediction)
    #             emotion_label_arg = np.argmax(emotion_prediction)
    #             emotion_text = self.emotion_labels[emotion_label_arg]
    #             emotion_window.append(emotion_text)
    #
    #             if len(emotion_window) > emotion_recognition.frame_window:
    #                 emotion_window.pop(0)
    #             try:
    #                 emotion_mode = mode(emotion_window)
    #             except:
    #                 continue
    #             if emotion_text == 'angry':
    #                 color = emotion_probability * np.asarray((255, 0, 0))
    #             elif emotion_text == 'sad':
    #                 color = emotion_probability * np.asarray((0, 0, 255))
    #             elif emotion_text == 'happy':
    #                 color = emotion_probability * np.asarray((255, 255, 0))
    #             elif emotion_text == 'surprise':
    #                 color = emotion_probability * np.asarray((0, 255, 255))
    #             else:
    #                 color = emotion_probability * np.asarray((0, 255, 0))
    #
    #             color = color.astype(int)
    #             color = color.tolist()
    #
    #             draw_bounding_box(face_coordinates, rgb_image, color)
    #             draw_text(face_coordinates, rgb_image, emotion_mode,
    #                       color, 0, -45, 1, 1)
    #
    #         bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    #         cv2.imshow('window_frame', bgr_image)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             video_capture.release()
    #             cv2.destroyAllWindows()
    #             break

    def emotions(self, frame):
        # print('                ', self.emotion_window)
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detect_faces(self.face_detection_model, gray_image)

        for face_coordinates in faces:
            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_recognition.emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]

            try:
                gray_face = cv2.resize(gray_face, (64, 64))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)

            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = self.emotion_labels[emotion_label_arg]
            self.emotion_window.append(emotion_text)

            if len(self.emotion_window) > emotion_recognition.frame_window:
                self.emotion_window.pop(0)
            try:
                emotion_mode = mode(self.emotion_window)
            except:
                continue

            if emotion_text == 'angry':
                color = emotion_probability * np.asarray((255, 0, 0))
            elif emotion_text == 'sad':
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_text == 'happy':
                color = emotion_probability * np.asarray((255, 255, 0))
            elif emotion_text == 'surprise':
                color = emotion_probability * np.asarray((0, 255, 255))
            else:
                color = emotion_probability * np.asarray((0, 255, 0))

            color = color.astype(int)
            color = color.tolist()

            return emotion_mode, color

            # draw_bounding_box(face_coordinates, rgb_image, color)
            # draw_text(face_coordinates, rgb_image, emotion_mode,
            #           color, 0, -45, 1, 1)

        # bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)


FER = FacialExpressionRecognition(use_haar_cascade=True)
