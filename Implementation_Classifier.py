import os
import pickle

import math
import numpy as np
import scipy.io as sio
import tensorflow as tf
from sklearn import neighbors
from sklearn import svm

# from sklearn.datasets import dump_svmlight_file as libsvm
from libs.FaceNet import facenet

LOG_DIR = 'logs'
metadata = os.path.join(LOG_DIR, 'metadata.tsv')
path_matfile = "./debugging/emb_array.mat"
path_libsvm = "./debugging/libsvm.libsvm"

class Classifier():
    def __init__(self, mode="CLASSIFY", data_dir="", trained_model=""):
        self.mode = mode
        self.data_dir = data_dir
        self.trained_model = trained_model
        self.batch_size = 32
        self.image_size = 160
        self.ALLOWED_EXTENSION = {'png', 'jpg', 'jpeg'}

        # self.seed = 666
        # self.min_nof_images_per_class = 20
        # self.nof_train_images_per_class = 10

    def sk_SVM(self, classifier_path=""):
        with tf.Graph().as_default():
            with tf.Session() as sess:
                dataset = facenet.get_dataset(self.data_dir)
                paths, labels = facenet.get_image_paths_and_labels(dataset)

                print(f"Number of classes: {len(dataset)}")
                print(f"Number of images: {len(paths)}")

                facenet.load_model(self.trained_model)

                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]
                # print(embedding_size)

                # Run forward pass to calculate embeddings
                print('Calculating features for images')
                nof_images = len(paths)
                nof_batches_per_epoch = int(math.ceil(1.0 * nof_images / self.batch_size))
                emb_array = np.zeros((nof_images, embedding_size))
                for i in range(nof_batches_per_epoch):
                    start_index = i * self.batch_size
                    end_index = min((i + 1) * self.batch_size, nof_images)
                    paths_batch = paths[start_index:end_index]
                    images = facenet.load_data(paths_batch, False, False, self.image_size)
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

                classifier_filename_exp = os.path.expanduser(classifier_path)
                kernel = 'linear'
                C = 1.0

                model = svm.SVC(kernel=kernel, probability=True, decision_function_shape='ovo', C=C)

                print(f"Training SVC Classifier with kernel: {kernel}, C={C}")
                # model = svm.SVC(C=1000, gamma=0.001, kernel='rbf', probability=True)
                # model = svm.SVC(C=100, gamma=0.01, kernel='rbf', probability=True)
                # model = svm.SVC(C=10, gamma=0.01, kernel='linear', probability=True)

                model.fit(emb_array, labels)

                print(f"[INFO] Embedding shape: {emb_array.shape}, Nof classes: {len(dataset)}")
                print(model.fit_status_)
                # Create a list of class names
                class_names = [cls.name.replace('_', ' ') for cls in dataset]

                # Saving classifier model
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)

                # Save extracted embeddings
                sio.savemat(path_matfile, mdict={"embs": emb_array, "labels": labels})

    def knn_training(self, classifier_path=None, n_neighbors=None,
                  knn_algo='ball_tree', verbose=False):
        """
        :param classifier_path:
        :param n_neighbors: number of neighbors to weigh in classification. Chosen automatically if not specified
        :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
        :param verbose: verbosity of training
        :return: returns knn classifier that was trained on the given data.
        """

        with tf.Graph().as_default():
            with tf.Session() as sess:
                dataset = facenet.get_dataset(self.data_dir)
                paths, labels = facenet.get_image_paths_and_labels(dataset)

                print(f"Number of classes: {len(dataset)}")
                print(f"Number of images: {len(paths)}")

                facenet.load_model(self.trained_model)

                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]

                # Run forward pass to calculate embeddings
                print('Calculating features for images')
                nof_images = len(paths)
                nof_batches_per_epoch = int(math.ceil(1.0 * nof_images / self.batch_size))
                emb_array = np.zeros((nof_images, embedding_size))
                for i in range(nof_batches_per_epoch):
                    start_index = i * self.batch_size
                    end_index = min((i + 1) * self.batch_size, nof_images)
                    paths_batch = paths[start_index:end_index]
                    images = facenet.load_data(paths_batch, False, False, self.image_size)
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

                # Save extracted embeddings
                sio.savemat(path_matfile, mdict={"embs": emb_array, "labels": labels})


        # Determine how many neighbors to use for weighting in the kNN classifier
        if n_neighbors is None:
            n_neighbors = int(round(math.sqrt(len(emb_array))))
            if verbose:
                print("Chose n_neighbors automatically:", n_neighbors)

        # Create and train the KNN classifier
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
        knn_clf.fit(emb_array, labels)

        if classifier_path is not None:
            with open(classifier_path, 'wb') as f:
                pickle.dump(knn_clf, f)

        # return knn_clf


class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


