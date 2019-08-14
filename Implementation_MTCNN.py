import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import misc

from libs.Emotions import emotion_recognition
from libs.FaceNet import facenet
from libs.MTCNN import detect_face


class MTCNN():
    def __init__(self):
        self.min_face_size = 50  # minimum size of face
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.scale_factor = 0.709  # scale factor

        self.emotion_labels = emotion_recognition.emotion_labels
        # Load TF default Graph and Session:
        gpu_memory_fraction = 1.0
        print('[MTCNN] Loading pre-trained P-Net R-Net O-Net')
        with tf.Graph().as_default():
            # gpu_options = tf.GPUOptions(
            #     per_process_gpu_memory_fraction=gpu_memory_fraction)
            # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
            #                                         log_device_placement=False))
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
            with sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, None)

                self.emotion_classifier = tf.keras.models.load_model(filepath=emotion_recognition.emotion_model_path,
                                                                     custom_objects=None, compile=True)


    def align_images_in_dir(self, data_dir):
        data_dir = data_dir
        images = os.listdir(data_dir)
        # print(images)
        for image in images:
            img = misc.imread(os.path.join(data_dir, image))
            bounding_boxes, _ = detect_face.detect_face(img, minsize=self.min_face_size,
                                                              pnet=self.pnet, rnet=self.rnet, onet=self.onet,
                                                              threshold= self.threshold, factor=self.scale_factor)
            for (x1, y1, x2, y2, acc) in bounding_boxes:
                w = x2-x1
                h = y2-y1

                cv.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
                # print('Accuracy score', acc)
                print('Top Left: ', x1,', ', y1)
                print('Bottom Right', x2, ', ', y2)

            # Save to file
            # misc.imsave('faceBoxed'+i, img)

            # Show the face with bounding box
            plt.figure()
            plt.imshow(img)
            plt.show()

    def live_camera(self):

        # MTCNN implementation with live webcam
        cv.namedWindow ( 'LiveCamera' ,cv.WINDOW_AUTOSIZE)
        cap = cv.VideoCapture ( 0 )

        while cap.isOpened ( ):
            ret , frame = cap.read ( )


            # Get Tick value
            start_time = cv.getTickCount()

            #   run detect_face from the facenet library
            bounding_boxes, _ = detect_face.detect_face(frame, minsize=self.min_face_size,
                                                              pnet=self.pnet, rnet=self.rnet, onet=self.onet,
                                                              threshold= self.threshold, factor=self.scale_factor)
            print(bounding_boxes)
            #   for each box
            for (x1, y1, x2, y2, acc) in bounding_boxes:

                w = x2-x1
                h = y2-y1
                #   plot the box using cv
                cv.rectangle(frame,(int(x1),int(y1)),(int(x1+w),int(y1+h)),(0,255,0),2)

            # Calculate FPS manually
            fps = str(abs(int(cv.getTickFrequency()/(cv.getTickCount()-start_time))))

            # With videos could use this
            # fps = str(cap.get(cv.CAP_PROP_FPS))

            font = cv.FONT_HERSHEY_SIMPLEX
            frameHeight, frameWidth = frame.shape[:2]
            cv.putText(frame,"FPS = "+ fps,(50,50), font, 0.5,(255,255,255),2,cv.LINE_AA)

            cv.imshow ( 'LiveCamera' , frame)
            k = cv.waitKey(1) & 0xFF
            if k == ord('q'):
                cap.release()
                cv.destroyAllWindows()

    def dataset_alignment(self, input_dir, output_dir):
        output_dir = os.path.expanduser(output_dir)
        nrof_total_images = 0
        nrof_successfully_aligned = 0
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dataset = facenet.get_dataset(input_dir)
        with tf.Graph().as_default():
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
            # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
            #                                         log_device_placement=False))
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
            with sess.as_default():
                # pnet, rnet, onet = detect_face.create_mtcnn(sess, "")
                pnet, rnet, onet = detect_face.create_mtcnn(sess, model_path="")

        min_face_size = 20
        threshold = [0.6, 0.7, 0.7]
        scale_factor = 0.709
        margin = 44
        image_size = 160
        # random_key = np.random.randint(0, high=9999)
        # bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
        # with open(bounding_boxes_filename, "w") as text_file:
        for cls in dataset:
            output_class_dir = os.path.join(output_dir, cls.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
            for image_path in cls.image_paths:
                nrof_total_images += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir, filename + '.png')
                print("Image Path: ", image_path)
                if not os.path.exists(output_filename):
                    try:
                        img = misc.imread(image_path)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim < 2:
                            print('Unable to align "%s"' % image_path)
                            # text_file.write('%s\n' % (output_filename))
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                            print('to_rgb data dimension: ', img.ndim)
                        img = img[:, :, 0:3]

                        bounding_boxes, _ = detect_face.detect_face(img, min_face_size, pnet, rnet, onet, threshold,
                                                                    scale_factor)
                        nrof_faces = bounding_boxes.shape[0]
                        # print('No of Detected Face: %d' % nrof_faces)
                        if nrof_faces > 0:
                            det = bounding_boxes[:, 0:4]
                            img_size = np.asarray(img.shape)[0:2]
                            if nrof_faces >= 1:
                                bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                                img_center = img_size / 2
                                offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                     (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                                index = np.argmax(
                                    bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                                det = det[index, :]
                            det = np.squeeze(det)
                            bb_temp = np.zeros(4, dtype=np.int32)

                            bb_temp[0] = max(det[0],0)
                            bb_temp[1] = max(det[1],0)
                            bb_temp[2] = max(det[2],0)
                            bb_temp[3] = max(det[3],0)

                            cropped_temp = img[bb_temp[1]:bb_temp[3], bb_temp[0]:bb_temp[2], :]
                            scaled_temp = misc.imresize(cropped_temp, (image_size, image_size), interp='bilinear')

                            nrof_successfully_aligned += 1
                            misc.imsave(output_filename, scaled_temp)
                            # text_file.write('%s %d %d %d %d\n' % (
                            #     output_filename, bb_temp[0], bb_temp[1], bb_temp[2], bb_temp[3]))
                        else:
                            print('Unable to align "%s"' % image_path)
                            # text_file.write('%s\n' % (output_filename))

        return nrof_total_images, nrof_successfully_aligned


MTCNN = MTCNN()

# MTCNN.live_camera()


# # MTCNN.align_images_in_dir(data_dir='./datasets/avengers_actors/raw/Chris_Evans')
#
# # for N in range(1, 4):
# #     MTCNN.dataset_alignment("/home/vantuan5644/PycharmProjects/DeepLearning/FaceNet/datasets/avengers_actors/raw",
# #                         output_dir="./datasets/avengers_actors/aligned")
# input_datadir = "/home/vantuan5644/PycharmProjects/DeepLearning/FaceNet/datasets/avengers_actors/raw"
# output_datadir = "/home/vantuan5644/PycharmProjects/DeepLearning/FaceNet/datasets/avengers_actors/aligned"
#
# nrof_images_total, nrof_successfully_aligned = MTCNN.dataset_alignment(input_datadir, output_datadir)
#
# print('Total number of images: %d' % nrof_images_total)
# print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

