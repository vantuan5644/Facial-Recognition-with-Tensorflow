import datetime
import os
from shutil import copyfile

import math
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from libs.FaceNet import facenet


class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)

class DatasetPrep:

    def __init__(self, dataset_path=None, split=False):
        self.dataset_path = dataset_path
        self.split = split
        self.dataset = facenet.get_dataset(self.dataset_path)

    def load_dataset(self):
        paths, labels = facenet.get_image_paths_and_labels(self.dataset)
        print(f"[Dataset] Number of classes: {len(self.dataset)}")
        print(f"[Dataset] Number of images: {len(paths)}")

    def split_dataset(self, split_ratio, min_nof_images_per_class):
        if self.split:

            train_set = []
            test_set = []
            for cls in self.dataset:
                paths = cls.image_paths
                np.random.shuffle(paths)
                nof_images_in_class = len(paths)
                split = int(math.floor(nof_images_in_class * (1 - split_ratio)))
                if split == nof_images_in_class:
                    split = nof_images_in_class -1
                if split >= min_nof_images_per_class and nof_images_in_class - split >= 1:
                    train_set.append(ImageClass(cls.name, paths[:split]))
                    test_set.append(ImageClass(cls.name, paths[split:]))

        return train_set, test_set

"""
Splitting the dataset into train-set and test-set
"""
split = DatasetPrep(dataset_path="/home/vantuan5644/PycharmProjects/DeepLearning/SmartCameraWithDeepLearning/datasets/raw",
                    split=True)
train, test = split.split_dataset(split_ratio=0.1, min_nof_images_per_class=3)

splitted_dir = "./datasets/splitted_dataset"

for cls in train:
    cls_name = cls.name

    if not os.path.exists(splitted_dir):
        os.makedirs(splitted_dir)

    cls_path = os.path.join(splitted_dir, "train", cls_name)
    if not os.path.exists(cls_path):
        os.makedirs(cls_path)

    for filepath in cls.image_paths:
        basename = os.path.basename(filepath)
        dst = os.path.join(cls_path, basename)
        copyfile(filepath, dst)
for cls in test:
    cls_name = cls.name

    if not os.path.exists(splitted_dir):
        os.makedirs(splitted_dir)

    cls_path = os.path.join(splitted_dir, "test", cls_name)
    if not os.path.exists(cls_path):
        os.makedirs(cls_path)

    for filepath in cls.image_paths:
        basename = os.path.basename(filepath)
        dst = os.path.join(cls_path, basename)
        copyfile(filepath, dst)


"""
Training the linear SVC classifier
"""


class CustomLinearSVC:

    def __init__(self, use_splitted_dataset=False, train_set=None, test_set=None, filename=None, path=None):

        if not use_splitted_dataset and test_set is None:
            self.training_set = facenet.get_dataset(train_set)
        elif use_splitted_dataset and train_set and test_set:
            self.training_set = facenet.get_dataset(train_set)
            self.test_set = facenet.get_dataset(test_set)

        self.paths, self.labels = facenet.get_image_paths_and_labels(self.training_set)

        print(f"[Dataset] Number of classes: {len(self.training_set)}")
        print(f"[Dataset] Number of images: {len(self.paths)}")

        if os.path.exists(os.path.join(path, filename)):
            date_time = str(datetime.datetime.now()).replace(" ","_")
            filename = filename.replace(".pkl","") + date_time + ".pkl"

        self.filepath = os.path.join(path, filename)
        self.image_size = 160

        self.emb_array = []

    def generate_embeddings(self, model_path=None, path_emb=None, path_mat_file=None, batch_size=64):
        if model_path is None:
            print(f"[ERROR] Specify the feature extractor!")

        elif path_emb is None:
            with tf.Graph().as_default():
                with tf.Session() as sess:

                    print("[INFO] Calculating faces encodings")
                    facenet.load_model(model_path)

                    # Get input and output tensors
                    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                    embedding_size = embeddings.get_shape()[1]

                    nof_images = len(self.paths)
                    nof_batches_per_epoch = int(math.ceil(1.0 * nof_images / batch_size))
                    emb_array = np.zeros((nof_images, embedding_size))
                    for i in range(nof_batches_per_epoch):
                        start_index = i * batch_size
                        end_index = min((i + 1) * batch_size, nof_images)
                        paths_batch = self.paths[start_index:end_index]
                        images = facenet.load_data(paths_batch, False, False, self.image_size)
                        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                        emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

                    self.emb_array = emb_array
                    print(f"[CLASSIFIER] {self.emb_array.shape}")

    def custom_linear_svc(self):

        # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
        #                      'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
        #                     {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
        #                      'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
        #                     {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
        #                     ]
        tuned_parameters = [{'kernel': ['linear'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                             'C': [0.001, 0.10, 0.1, 1., 10, 25, 50, 100, 1000]},
                            {'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                             'C': [0.001, 0.10, 0.1, 1, 10, 25, 50, 100, 1000]}]

        scores = ['precision', 'recall']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=8)
            clf.fit(self.emb_array, self.labels)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()


tuned_svc = CustomLinearSVC(use_splitted_dataset=False,
                            train_set="./datasets/aligned",
                            test_set=None,
                            filename="CustomLinearSVC.pkl",
                            path="./models/SVM")

model_path = "./models/facenet/20180402-114759/20180402-114759.pb"
# model_path = "./models/facenet/20180408-102900/20180408-102900.pb"
tuned_svc.generate_embeddings(model_path=model_path,
                              path_emb=None,
                              path_mat_file=None)

tuned_svc.custom_linear_svc()