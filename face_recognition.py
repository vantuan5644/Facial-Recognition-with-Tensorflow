from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import numpy as np
import tensorflow as tf

from Emotions import FER
from Implementation_Classifier import Classifier
from Implementation_MTCNN import MTCNN
from Multithreading import *
from database_management import pdCSV
from libs.FaceNet import facenet
from libs.MTCNN import detect_face


# ----------------------------------------------

class FaceRecognition():

    def __init__(self, dataset_base_dir=None,
                 classifier_base_dir=None,
                 classifier_name=None,
                 model_path=None,
                 haar_cascade=None):

        # Declare directories
        self.raw_dir = os.path.join(dataset_base_dir, "raw")
        self.aligned_dir = os.path.join(dataset_base_dir, "aligned")

        for dirs in (self.raw_dir, self.aligned_dir):
            if not os.path.exists(dirs):
                os.mkdir(dirs)

        self.classifier_path = os.path.join(classifier_base_dir, classifier_name)
        # SVM Classifier
        self.linearSVM = Classifier(mode="CLASSIFY",
                                    data_dir=self.aligned_dir,
                                    trained_model=model_path)
        # Haar Cascade Face Detector
        self.haar_cascade_xml = haar_cascade

        # Tensorflow inference
        self.graph = tf.get_default_graph()
        with self.graph.as_default():
            if tf.test.is_gpu_available():
                config = tf.ConfigProto(log_device_placement=False)
                config.gpu_options.allow_growth = True
                self.sess = tf.Session(config=config)
            else:
                print('[HW INFO] Detected no GPU, executing in CPU mode')
                self.sess = tf.Session()

        # MTCNN Subnets
        self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess,
                                                                   model_path="")
        print("[INFO] MTCNN has been loaded into RAM")

        # FaceNet
        facenet.load_model(model_path)
        self.facenet_images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.facenet_embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.facenet_phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        self.facenet_embedding_size = self.facenet_embeddings.get_shape()[1]
        print("[INFO] FaceNet has been loaded into RAM")

        # SVM
        classifier_filename_exp = os.path.expanduser(self.classifier_path)
        with open(classifier_filename_exp, 'rb') as f:
            (self.svm_model, self.svm_labels) = pickle.load(f)
        print("[INFO] SVM Classifier Extracted")

        print(f"[Classifier] Recognized {len(self.svm_labels)} people: {self.svm_labels}")

    def align_images_in_dataset(self):

        # Aligning images in dataset

        (nof_images_total, nof_successfully_aligned) = MTCNN.dataset_alignment(self.raw_dir,
                                                                               self.aligned_dir)
        print('Total number of images: %d' % nof_images_total)
        print('Number of successfully aligned images: %d' % nof_successfully_aligned)

        # Train a new custom classifier
        self.linearSVM.sk_SVM(classifier_path=self.classifier_path)

    def add_new_user(self, new_user_name, src):

        i = 0
        tmp_user_name = new_user_name

        while os.path.exists(os.path.join(self.raw_dir, tmp_user_name)):
            i += 1
            tmp_user_name = f"{new_user_name} ({i})"
            print(tmp_user_name)

        new_user_path = os.path.join(self.raw_dir, tmp_user_name)
        os.mkdir(new_user_path)
        # OpenCV Capturing
        cap = cv2.VideoCapture(src)
        face_detector = cv2.CascadeClassifier(self.haar_cascade_xml)
        # For each person, enter one numeric face id
        # new_user_id = input('\n enter user id end press <return> ==>  ')
        print("\n [INFO] Initializing face capture. Look the camera and wait ...")
        # Initialize individual sampling face count
        count = 0
        while True:
            ret, img = cap.read()
            # print('Start Capturing')
            img = cv2.flip(img, 1)  # flip video image vertically
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            if count < 30:
                for (x, y, w, h) in faces:
                    cv2.imwrite(new_user_path + '/' + tmp_user_name + '_' + str(count) + ".png", img)
                    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    count += 1
            cv2.imshow('capturing', img)

            k = cv2.waitKey(1) & 0xff
            if k == ord('q') or count == 30:
                break

        pdCSV.add_new_user(name=new_user_name)

        print("\n [INFO] Exiting Program and cleaning up stuff")
        cap.release()
        cv2.destroyAllWindows()

    # TODO 1. Explain MTCNN hyperparams minsize, threshold, scalefactor
    # TODO 2. What's pre-whiten
    # TODO 3. Save the processed embeddings to files
    # TODO 4. Bounding box text? What for?
    # TODO 5. Improving reliability

    def image_test(self, img_path="", emotions=False):

        cv2.namedWindow('test', cv2.WINDOW_NORMAL)
        frame = cv2.imread(img_path)

        #
        # # Do fixed image standardization
        # img = (img - 127.5) / 128.0

        # MTCNN implementation
        total_boxes, _ = detect_face.detect_face(frame,
                                                 minsize=20,
                                                 pnet=self.pnet, rnet=self.rnet, onet=self.onet,
                                                 threshold=[.6, .7, .7], factor=.709)

        # print(total_boxes)
        nof_faces = total_boxes.shape[0]

        color = [(255, 0, 0),
                 (0, 255, 0),
                 (0, 0, 255),
                 (255, 255, 0),
                 (255, 0, 255),
                 (0, 255, 255),
                 ]

        if nof_faces > 0:
            img_size = np.asarray(frame.shape)[0:2]
            bounding_boxes = total_boxes[:, 0:4]
            # contains the bounding_boxes coordinates only

            bounding_box = np.zeros((nof_faces, 4), dtype=np.int32)

            cropped = []
            scaled = []
            scaled_reshape = []

            for i in range(nof_faces):
                emb_array = np.zeros((1, self.facenet_embedding_size))

                bounding_box[i][0] = bounding_boxes[i][0]
                bounding_box[i][1] = bounding_boxes[i][1]
                bounding_box[i][2] = bounding_boxes[i][2]
                bounding_box[i][3] = bounding_boxes[i][3]

                # Draw the bounding box
                bb_color = color[i]
                cv2.rectangle(frame, (bounding_box[i][0], bounding_box[i][1]),
                              (bounding_box[i][2], bounding_box[i][3]), bb_color, 2)

                # Warning if someone standing too close to the camera
                if bounding_box[i][0] <= 0 \
                        or bounding_box[i][1] <= 0 \
                        or bounding_box[i][2] >= len(frame[0]) \
                        or bounding_box[i][3] >= len(frame):
                    print("[WARNING] You're standing too close to the camera!")
                    continue

                # Cropping and aligning
                cropped.append(frame[bounding_box[i][1]:bounding_box[i][3],
                               bounding_box[i][0]:bounding_box[i][2], :])

                # cv2.namedWindow(str(i), cv2.WINDOW_NORMAL)
                # cv2.imshow(str(i), cropped[i])
                # cropped[j] = facenet.flip(cropped[j], False)

                # gray_face = cv2.cvtColor(cropped[j], cv2.COLOR_BGR2GRAY)
                # cv2.namedWindow('gray',cv2.WINDOW_NORMAL)
                # cv2.imshow('gray', gray_face)

                input_image_size = 160
                # scaled.append(misc.imresize(cropped[j], (input_image_size, input_image_size), interp='bilinear'))
                scaled.append(cv2.resize(cropped[i], (input_image_size, input_image_size)))
                scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size), interpolation=cv2.INTER_CUBIC)

                # Pre-whitening
                # Subtracting the average and normalizes the range of the pixel values of input images.
                scaled[i] = facenet.prewhiten(scaled[i])
                scaled_reshape.append(scaled[i].reshape(1, input_image_size, input_image_size, 3))

                # Run feedforward to calculate the embedding
                feed_dict = {self.facenet_images_placeholder: scaled_reshape[i],
                             self.facenet_phase_train_placeholder: False}
                emb_array[0, :] = self.sess.run(self.facenet_embeddings,
                                                feed_dict=feed_dict)

                # predictions = self.svm_model.predict_proba(emb_array)
                # ndarray
                # [[0.1 0.2 0.3]]

                # best_class_index = np.argmax(predictions, axis=1)
                # e.g [2]

                # best_class_probability = predictions[np.arange(len(best_class_index)), best_class_index]
                # return the value of argmax index
                # ([0.3])
                # np.save("./debugging/Tuan.npy",emb_array)

                predict = self.svm_model.predict(emb_array)
                # [62]
                predict_index = int(predict)
                # 62
                predict_label = list(self.svm_labels)[predict_index]
                # Tuan Tran
                prob_score_array = self.svm_model.predict_proba(emb_array)

                pos_x = bounding_box[i][0]
                pos_y = bounding_box[i][3] + 20

                unknown_case = True
                if np.amax(prob_score_array) >= 0.1:
                    unknown_case = False
                    predict_index_prob = np.argmax(prob_score_array)
                    if int(predict_index_prob) == int(predict_index):
                        final_result = predict_label
                        # print(f"[Detected] Result: {final_result}")
                        cv2.putText(frame, final_result, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                                    thickness=2, lineType=2)
                if unknown_case:
                    print(f"[Unknown] Highest-score-class: {predict_label}")

                if emotions:
                    try:
                        emotion, color = FER.emotions(frame)
                        cv2.putText(frame, emotion, (pos_x + 150, pos_y),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, thickness=2, color=color)
                    except:
                        continue
                else:
                    continue
        cv2.imshow('test', frame)
        cv2.waitKey(0)

    def live_camera(self, src, emotions=True):

        # src = 'http://192.168.137.85:4747/mjpegfeed?640x480'
        # For multi-threading optimization
        video_getter = VideoGet(src).start()
        video_show = VideoShow(video_getter.frame).start()
        cps = CountsPerSec.start()

        while True:
            if video_getter.stopped or video_show.stopped:
                video_getter.stop()
                video_show.stop()
                break

            frame = video_getter.frame
            # frame = cv2.resize(frame, dsize=(), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            if frame.ndim == 2:
                frame = facenet.to_rgb(frame)

            frame = frame[:, :, 0:3]

            # Get SysTick timer's value to calculate FPS manually
            start_time = cv2.getTickCount()

            # MTCNN hyperparams
            # minsize: minimum of faces' size
            # pnet, rnet, onet: caffe-model
            # threshold: threshold=[th1 th2 th3], th1-3 are three steps's threshold
            # scale factor in image pyramid
            total_boxes, _ = detect_face.detect_face(frame,
                                                     minsize=80,
                                                     pnet=self.pnet,
                                                     rnet=self.rnet,
                                                     onet=self.onet,
                                                     threshold=[.6, .7, .7], factor=.709)

            # print(total_boxes)
            nof_faces = total_boxes.shape[0]

            color = [(255, 0, 0),
                     (0, 255, 0),
                     (0, 0, 255),
                     (255, 255, 0),
                     (255, 0, 255),
                     (0, 255, 255),
                     ]
            result_dict = {"nof_faces": nof_faces,
                           "recognized": False,
                           "who": None,
                           "highest_score_class": None,
                           "fer": None,
                           "warnings": None}
            if nof_faces == 0:
                print("[WARNING] Can't find anybody here")
                result_dict['nof_faces'] = 0
            if nof_faces > 0:
                img_size = np.asarray(frame.shape)[0:2]
                bounding_boxes = total_boxes[:, 0:4]
                # contains the bounding_boxes coordinates only

                bounding_box = np.zeros((nof_faces, 4), dtype=np.int32)

                cropped = []
                scaled = []
                scaled_reshape = []
                j = 0

                for i in range(nof_faces):
                    try:
                        emb_array = np.zeros((1, self.facenet_embedding_size))

                        bounding_box[i][0] = bounding_boxes[i][0]
                        bounding_box[i][1] = bounding_boxes[i][1]
                        bounding_box[i][2] = bounding_boxes[i][2]
                        bounding_box[i][3] = bounding_boxes[i][3]
                        # Draw the bounding box
                        bb_color = color[i]

                        cv2.rectangle(frame, (bounding_box[i][0], bounding_box[i][1]),
                                      (bounding_box[i][2], bounding_box[i][3]), bb_color, 2)

                        # Warning if someone standing too close to the camera
                        if bounding_box[i][0] <= 0 \
                                or bounding_box[i][1] <= 0 \
                                or bounding_box[i][2] >= len(frame[0]) \
                                or bounding_box[i][3] >= len(frame):
                            print("[WARNING] You're standing too close to the camera!")
                            result_dict['warnings'] = "Face Overlapped"
                            continue

                        cropped.append(frame[bounding_box[i][1]:bounding_box[i][3],
                                       bounding_box[i][0]:bounding_box[i][2], :])
                        cropped[0] = facenet.flip(cropped[0], False)

                        # gray_face = cv2.cvtColor(cropped[i], cv2.COLOR_BGR2GRAY)
                        # cv2.namedWindow('gray',cv2.WINDOW_NORMAL)
                        # cv2.imshow('gray', gray_face)

                        input_image_size = 160
                        # scaled.append(misc.imresize(cropped[j], (input_image_size, input_image_size), interp='bilinear'))a
                        scaled.append(cv2.resize(cropped[0], (input_image_size, input_image_size)))
                        scaled[0] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                               interpolation=cv2.INTER_CUBIC)

                        # Pre-whitening
                        # Subtracting the average and normalizes the range of the pixel values of input images.
                        scaled[0] = facenet.prewhiten(scaled[0])
                        scaled_reshape.append(scaled[0].reshape(1, input_image_size, input_image_size, 3))

                        # Run feedforward to calculate the embedding
                        feed_dict = {self.facenet_images_placeholder: scaled_reshape[i],
                                     self.facenet_phase_train_placeholder: False}
                        emb_array[0, :] = self.sess.run(self.facenet_embeddings, feed_dict=feed_dict)

                        # predictions = self.svm_model.predict_proba(emb_array)
                        # # ndarray
                        # # [[0.1 0.2 0.3]]
                        #
                        # best_class_index = np.argmax(predictions, axis=1)
                        # # e.g [2]
                        #
                        # best_class_probability = predictions[np.arange(len(best_class_index)), best_class_index]
                        # # return the value of argmax index
                        # # ([0.3])
                        # # np.save("./debugging/Tuan.npy",emb_array)

                        predict = self.svm_model.predict(emb_array)
                        # [62]
                        predict_index = int(predict)
                        # 62
                        predict_label = list(self.svm_labels)[predict_index]
                        # Tuan Tran
                        prob_score_array = self.svm_model.predict_proba(emb_array)

                        pos_x = bounding_box[i][0]
                        pos_y = bounding_box[i][3] + 20

                        unknown_case = True
                        if np.amax(prob_score_array) >= 0.1:
                            unknown_case = False
                            predict_index_prob = np.argmax(prob_score_array)
                            if int(predict_index_prob) == int(predict_index):
                                final_result = predict_label
                                cv2.putText(frame, final_result, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (0, 0, 255),
                                            thickness=1, lineType=2)
                                # print(f"[Detected] Result: {final_result}")
                                result_dict['recognized'] = True
                                result_dict['who'] = predict_label

                        if unknown_case:
                            # print(f"[Unknown] Highest-score-class: {predict_label}")
                            result_dict['recognized'] = False
                            result_dict['highest_score_class'] = predict_label
                            indices = (-prob_score_array).argsort()[:3]
                            # print(indices)
                            # for idx in indices:
                            #     print(list(self.svm_labels)[idx])

                        if emotions:
                            try:
                                emotion, color = FER.emotions(frame)
                                cv2.putText(frame, emotion, (pos_x, pos_y + 15),
                                            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1, color=color)
                                result_dict['fer'] = emotion
                            except:
                                continue

                        pdCSV.cases_handling(result_dict=result_dict)


                    except:
                        continue

                    # j += 1
                # Calculate FPS manually
                fps = str(abs(int(cv2.getTickFrequency() / (cv2.getTickCount() - start_time))))
                cv2.putText(frame, "FPS = " + fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                            cv2.LINE_AA)

                frame = putIterationsPerSec(frame, cps.countsPerSec())
                video_show.frame = frame

                cps.increment()
                # print(result_dict)

    def recognize(self, frame, emotions=True, show_fps=True):
        """
        :param frame: Detect faces in one video frame, select camera source in somewhere else
        :param emotions: enable facial emotion recognition functionality
        :param show_fps: enable FPS calculation
        :return: face detection result & FER (if enabled)
        """

        if frame.ndim == 2:
            frame = facenet.to_rgb(frame)

        frame = frame[:, :, 0:3]

        # Get SysTick timer's value to calculate FPS manually
        start_time = cv2.getTickCount()

        # MTCNN hyperparams
        # minsize: minimum of faces' size
        # pnet, rnet, onet: caffe-model
        # threshold: threshold=[th1 th2 th3], th1-3 are three steps's threshold
        # scale factor in image pyramid
        total_boxes, _ = detect_face.detect_face(frame,
                                                 minsize=20,
                                                 pnet=self.pnet,
                                                 rnet=self.rnet,
                                                 onet=self.onet,
                                                 threshold=[.6, .7, .7], factor=.709)

        # print(total_boxes)
        nof_faces = total_boxes.shape[0]

        color = [(255, 0, 0),
                 (0, 255, 0),
                 (0, 0, 255),
                 (255, 255, 0),
                 (255, 0, 255),
                 (0, 255, 255),
                 ]
        result_dict = {"nof_faces": 0,
                       "recognized": False,
                       "who": None,
                       "highest_score_class": None,
                       "fer": None,
                       "warnings": None}
        if nof_faces == 0:
            print("[WARNING] Can't find anybody here")
            return frame, result_dict

        if nof_faces > 0:
            result_dict['nof_faces'] = nof_faces
            img_size = np.asarray(frame.shape)[0:2]
            bounding_boxes = total_boxes[:, 0:4]
            # contains the bounding_boxes coordinates only

            bounding_box = np.zeros((nof_faces, 4), dtype=np.int32)

            cropped = []
            scaled = []
            scaled_reshape = []
            j = 0

            for i in range(nof_faces):
                try:
                    emb_array = np.zeros((1, self.facenet_embedding_size))

                    bounding_box[i][0] = bounding_boxes[i][0]
                    bounding_box[i][1] = bounding_boxes[i][1]
                    bounding_box[i][2] = bounding_boxes[i][2]
                    bounding_box[i][3] = bounding_boxes[i][3]
                    # Draw the bounding box
                    bb_color = color[i]

                    cv2.rectangle(frame, (bounding_box[i][0], bounding_box[i][1]),
                                  (bounding_box[i][2], bounding_box[i][3]), bb_color, 2)

                    # Warning if someone standing too close to the camera
                    if bounding_box[i][0] <= 0 \
                            or bounding_box[i][1] <= 0 \
                            or bounding_box[i][2] >= len(frame[0]) \
                            or bounding_box[i][3] >= len(frame):
                        print("[WARNING] You're standing too close to the camera!")
                        result_dict['warnings'] = "Face Overlapped"
                        continue

                    cropped.append(frame[bounding_box[i][1]:bounding_box[i][3],
                                   bounding_box[i][0]:bounding_box[i][2], :])
                    cropped[0] = facenet.flip(cropped[0], False)

                    # gray_face = cv2.cvtColor(cropped[i], cv2.COLOR_BGR2GRAY)
                    # cv2.namedWindow('gray',cv2.WINDOW_NORMAL)
                    # cv2.imshow('gray', gray_face)

                    input_image_size = 160
                    # scaled.append(misc.imresize(cropped[j], (input_image_size, input_image_size), interp='bilinear'))a
                    scaled.append(cv2.resize(cropped[0], (input_image_size, input_image_size)))
                    scaled[0] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                           interpolation=cv2.INTER_CUBIC)

                    # Pre-whitening
                    # Subtracting the average and normalizes the range of the pixel values of input images.
                    scaled[0] = facenet.prewhiten(scaled[0])
                    scaled_reshape.append(scaled[0].reshape(1, input_image_size, input_image_size, 3))

                    # Run feedforward to calculate the embedding
                    feed_dict = {self.facenet_images_placeholder: scaled_reshape[i],
                                 self.facenet_phase_train_placeholder: False}
                    emb_array[0, :] = self.sess.run(self.facenet_embeddings, feed_dict=feed_dict)

                    # predictions = self.svm_model.predict_proba(emb_array)
                    # # ndarray
                    # # [[0.1 0.2 0.3]]
                    #
                    # best_class_index = np.argmax(predictions, axis=1)
                    # # e.g [2]
                    #
                    # best_class_probability = predictions[np.arange(len(best_class_index)), best_class_index]
                    # # return the value of argmax index
                    # # ([0.3])
                    # # np.save("./debugging/Tuan.npy",emb_array)

                    predict = self.svm_model.predict(emb_array)
                    # [62]
                    predict_index = int(predict)
                    # 62
                    predict_label = list(self.svm_labels)[predict_index]
                    # Tuan Tran
                    prob_score_array = self.svm_model.predict_proba(emb_array)

                    pos_x = bounding_box[i][0]
                    pos_y = bounding_box[i][3] + 20

                    unknown_case = True
                    if np.amax(prob_score_array) >= 0.1:
                        unknown_case = False
                        predict_index_prob = np.argmax(prob_score_array)
                        if int(predict_index_prob) == int(predict_index):
                            final_result = predict_label
                            cv2.putText(frame, final_result, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                                        thickness=1, lineType=2)
                            # print(f"[Detected] Result: {final_result}")
                            result_dict['recognized'] = True
                            result_dict['who'] = predict_label

                    if unknown_case:
                        # print(f"[Unknown] Highest-score-class: {predict_label}")
                        result_dict['recognized'] = False
                        result_dict['highest_score_class'] = predict_label
                        indices = (-prob_score_array).argsort()[:3]
                        # print(indices)
                        # for idx in indices:
                        #     print(list(self.svm_labels)[idx])

                    if emotions:
                        try:
                            emotion, color = FER.emotions(frame)
                            cv2.putText(frame, emotion, (pos_x, pos_y + 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1, color=color)
                            result_dict['fer'] = emotion
                        except:
                            continue

                except:
                    continue

                # j += 1
            # Calculate FPS manually
            if show_fps:
                fps = str(abs(int(cv2.getTickFrequency() / (cv2.getTickCount() - start_time))))
                cv2.putText(frame, "FPS = " + fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                            cv2.LINE_AA)

        return frame, result_dict


FaceRecognition = FaceRecognition(dataset_base_dir="./datasets",
                                  classifier_name="SVM.pkl",
                                  model_path="./models/facenet/20180402-114759/20180402-114759.pb",
                                  haar_cascade='./libs/haarcascade_frontalface_default.xml',
                                  classifier_base_dir="./models/SVM"
                                  )

# FaceRecognition.image_test("./test_img/test.png")

# FaceRecognition.add_new_user("TuanTran", src=0)
# FaceRecognition.align_images_in_dataset()

# if __name__ == "__main__":
FaceRecognition.live_camera(src=0, emotions=True)

# TODO: Lam lai mo hinh nhan dien cam xuc (tuan sau)
# TODO: Trinh bay bao cao LV
# TODO: viet chuogn trinfh huan luyen khuon mat moi
