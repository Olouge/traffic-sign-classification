import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os
import csv

from sklearn.preprocessing import LabelBinarizer

from serializers.trained_data_serializer import TrainedDataSerializer


class GermanTrafficSignDataset:
    """
    This class contains the following mechanisms:

       1. #configure - Trains a new network from the original traffic sign dataset
       2. #persist   - Saves a training, validation and test set after training a network
       3. #restore   - Restores a serialized training, validation and test set via to feed into another network
       4. Passing an instance of this class to print() prints some hueristics about the dataset.
    """

    def __init__(self):
        self.train_orig, self.validate_orig, self.test_orig = None, None, None
        self.train_gray, self.validate_gray, self.test_gray = None, None, None
        self.train_flat, self.validate_flat, self.test_flat = None, None, None

        self.train_labels, self.train_size, self.train_coords = None, None, None
        self.validate_labels, self.validate_size, self.validate_coords = None, None, None
        self.test_labels, self.test_size, self.test_coords = None, None, None

        self.num_training, self.num_validation, self.num_testing, self.num_classes = None, None, None, None

        # batch training metrics
        self._epochs_completed = 0
        self._index_in_epoch = 0

        self.__configured = False

    def configure(self, one_hot=True, train_validate_split_percentage=0.05):
        """
        Pipeline import sequence

          1. Loads the original data from the respective pickle files.
          2. Splits the training set into a training and validation set.
          3.
        """
        if not self.__configured:
            self.__one_hot_encoded = one_hot
            self.split_size = train_validate_split_percentage
            self.sign_names_map = self.__load_sign_names_map()

            [f() for f in [
                self.__load_data,
                self.__split_train_and_validation,
                self.__compute_metrics,
                self.__prepare_images,
                self.__one_hot_encode_labels
            ]]
            self.__configured = True

    def restore(self, pickle_file='trafficsigns_trained.pickle'):
        """
        Pipeline import sequence

          1. Loads the original data from the respective pickle files.
          2. Splits the training set into a training and validation set.
          3.
        """
        self.__restore(pickle_file)
        if not self.__configured:
            [f() for f in [
                self.__compute_metrics
            ]]
            self.__configured = True

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.num_training:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self.num_training)
            np.random.shuffle(perm)
            self.train_flat = self.train_flat[perm]
            self.train_labels = self.train_labels[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self.num_training
        end = self._index_in_epoch
        return self.train_flat[start:end], self.train_labels[start:end], start, end

    def serialize(self, data={}):
        return {**data, **{
            'sign_names_map': self.sign_names_map,

            'train_orig': self.train_orig,
            'validate_orig': self.validate_orig,
            'test_orig': self.test_orig,

            'train_gray': self.train_gray,
            'validate_gray': self.validate_gray,
            'test_gray': self.test_gray,

            'train_flat': self.train_flat,
            'validate_flat': self.validate_flat,
            'test_flat': self.test_flat,

            'train_labels': self.train_labels,
            'validate_labels': self.validate_labels,
            'test_labels': self.test_labels,

            'one_hot': self.__one_hot_encoded,
            'split_size': self.split_size
        }}

    def persist(self, data, pickle_file='trafficsigns_trained.pickle', overwrite=False):
        if self.__configured:
            TrainedDataSerializer.save_data(
                data=data,
                pickle_file=pickle_file,
                overwrite=overwrite
            )

    def __restore(self, pickle_file='trafficsigns_trained.pickle'):
        if not self.__configured:
            data = TrainedDataSerializer.reload_data(pickle_file=pickle_file)

            self.__one_hot_encoded = data['one_hot']
            self.split_size = data['split_size']
            self.sign_names_map = data['sign_names_map']

            self.train_orig, self.validate_orig, self.test_orig = data['train_orig'], data['validate_orig'], data[
                'test_orig']
            self.train_gray, self.validate_gray, self.test_gray = data['train_gray'], data['validate_gray'], data[
                'test_gray']
            self.train_flat, self.validate_flat, self.test_flat = data['train_flat'], data['validate_flat'], data[
                'test_flat']
            self.train_labels, self.validate_labels, self.test_labels = data['train_labels'], data['validate_labels'], \
                                                                        data['test_labels']
            del data
            print('train features shape: ', self.train_orig.shape)

    def plot_images(self):
        if self.__configured:
            ImagePlotter.plot_images(ImageTransformer.jitter_images(self.train_orig[:20]), self.train_labels[:20])
            ImagePlotter.plot_images(self.train_gray[:20], self.train_labels[:20], cmap='gray')

            ImagePlotter.plot_images(ImageTransformer.jitter_images(self.test_orig[:20]), self.test_labels[:20])
            ImagePlotter.plot_images(self.test_gray[:20], self.test_labels[:20], cmap='gray')

            ImagePlotter.plot_images(ImageTransformer.jitter_images(self.validate_orig[:20]), self.validate_labels[:20])
            ImagePlotter.plot_images(self.validate_gray[:20], self.validate_labels[:20], cmap='gray')

    # private

    def __load_sign_names_map(self):
        map = {}
        sign_names_path = os.path.join(os.path.dirname(__file__), '..', 'signnames.csv')
        with open(sign_names_path, 'r') as sign_names:
            has_header = csv.Sniffer().has_header(sign_names.read(1024))
            sign_names.seek(0)  # rewind
            incsv = csv.reader(sign_names)
            if has_header:
                next(incsv)  # skip header row
            plots = csv.reader(sign_names, delimiter=',')
            for row in plots:
                map[int(row[0])] = str(row[1])
        return map

    def __load_data(self):
        """
        Loads in train features and labels and test features and labels from their respective pickle file
        """

        training_file = os.path.join(os.path.dirname(__file__), '..', 'traffic-sign-data', 'train.p')
        testing_file = os.path.join(os.path.dirname(__file__), '..', 'traffic-sign-data', 'test.p')

        with open(training_file, mode='rb') as f:
            train = pickle.load(f)
        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)

        self.train_orig, self.train_labels, self.train_size, self.train_coords = train['features'], train['labels'], \
                                                                                 train['sizes'], train['coords']

        self.test_orig, self.test_labels, self.test_size, self.test_coords = test['features'], test['labels'], \
                                                                             test['sizes'], test['coords']

        print('Loaded traffic-sign-data/train.p and traffic-sign-data/test.p')

    def __split_train_and_validation(self):
        """
        Get randomized datasets for training and validation
        """

        self.train_orig, self.validate_orig, self.train_labels, self.validate_labels = train_test_split(
            self.train_orig,
            self.train_labels,
            test_size=self.split_size,
            random_state=832224)

        print(
            'Training features and labels randomized and split with train_test_split (validation % of training set: {})'.format(
                self.split_size))

    def __compute_metrics(self):
        self.num_training = len(self.train_orig)
        self.num_validation = len(self.validate_orig)
        self.num_testing = len(self.test_orig)
        self.num_classes = len(np.unique(self.train_labels))

        print('Detected {} training features, {} validation features, {} test features and {} unique classes.'.format(
            self.num_training, self.num_validation, self.num_testing, self.num_classes))

    def __prepare_images(self):
        """
        This method bucketizes the original images.

            orig:       The original unprocessed images
            gray:       The original unprocessed images with a grayscale filter applied
            flat:       The grayscale images flattened into a vector
        """
        train_orig_images = []
        train_gray_images = []
        train_flat_images = []

        validate_orig_images = []
        validate_gray_images = []
        validate_flat_images = []

        test_orig_images = []
        test_gray_images = []
        test_flat_images = []

        for image in self.train_orig:
            gray_image = self.__color2gray(image)
            flat_image = np.array(gray_image, dtype=np.float32).flatten()

            train_orig_images.append(image)
            train_gray_images.append(gray_image)
            train_flat_images.append(flat_image)

        for image in self.validate_orig:
            gray_image = self.__color2gray(image)
            flat_image = np.array(gray_image, dtype=np.float32).flatten()

            validate_orig_images.append(image)
            validate_gray_images.append(gray_image)
            validate_flat_images.append(flat_image)

        for image in self.test_orig:
            gray_image = self.__color2gray(image)
            flat_image = np.array(gray_image, dtype=np.float32).flatten()

            test_orig_images.append(image)
            test_gray_images.append(gray_image)
            test_flat_images.append(flat_image)

        # orig bucket
        self.train_orig = np.array(train_orig_images)
        self.validate_orig = np.array(validate_orig_images)
        self.test_orig = np.array(test_orig_images)

        # gray bucket
        self.train_gray = np.array(train_gray_images)
        self.validate_gray = np.array(validate_gray_images)
        self.test_gray = np.array(test_gray_images)

        # flat bucket
        self.train_flat = self.__normalize_greyscale(train_flat_images)
        self.validate_flat = self.__normalize_greyscale(validate_flat_images)
        self.test_flat = self.__normalize_greyscale(test_flat_images)

        print(
            'Bucketized german traffic sign images into three buckets: orig, gray and flat. flat is ' \
            'used for network training while orig and gray are meant for visulizations.'
        )

    def __one_hot_encode_labels(self):
        if self.__one_hot_encoded:
            # [Adapted from Lesson 7 - MiniFlow]
            # Turn labels into numbers and apply One-Hot Encoding
            encoder = LabelBinarizer()
            encoder.fit(self.train_labels)
            self.train_labels = encoder.transform(self.train_labels)

            # encoder = LabelBinarizer()
            # encoder.fit(self.validate_labels)
            self.validate_labels = encoder.transform(self.validate_labels)

            # encoder = LabelBinarizer()
            # encoder.fit(self.test_labels)
            self.test_labels = encoder.transform(self.test_labels)

            # Change to float32, so it can be multiplied against the features in TensorFlow, which are float32
            self.train_labels = self.train_labels.astype(np.float32)
            self.validate_labels = self.validate_labels.astype(np.float32)
            self.test_labels = self.test_labels.astype(np.float32)

            print('train, validate and test labels have been one-hot encoded using LabelBinarizer.')

    def __color2gray(self, image):
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
        return gray

    def __normalize_greyscale(self, image_data):
        """
        Leveraged from Lesson 7 from Tensorflow lab.
        Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
        :param image_data: The image data to be normalized
        :return: Normalized image data
        """
        a = 0.1
        b = 0.9
        x_min = np.min(image_data)
        x_max = np.max(image_data)
        x_prime = [a + (((x - x_min) * (b - a)) / (x_max - x_min)) for x in image_data]
        return np.array(x_prime)

    def __str__(self):
        result = []
        result.append(' ')
        result.append('One-Hot Encoded:             {}'.format(self.__one_hot_encoded))
        result.append('Train/Validation Split %:    {}'.format(self.split_size))
        result.append(' ')
        result.append('Training size:               {}'.format(self.num_training))
        result.append('Validation size:             {}'.format(self.num_validation))
        result.append('Testing size:                {}'.format(self.num_testing))
        result.append('Total classes:               {}'.format(self.num_classes))
        result.append(' ')
        result.append('Training orig shape:         {}'.format(self.train_orig.shape))
        result.append('Training gray shape:         {}'.format(self.train_gray.shape))
        result.append('Training flat shape:         {}'.format(self.train_flat.shape))
        result.append(' ')
        result.append('Validation orig shape:       {}'.format(self.validate_orig.shape))
        result.append('Validation gray shape:       {}'.format(self.validate_gray.shape))
        result.append('Validation flat shape:       {}'.format(self.validate_flat.shape))
        result.append(' ')
        result.append('Testing orig shape:          {}'.format(self.test_orig.shape))
        result.append('Testing gray shape:          {}'.format(self.test_gray.shape))
        result.append('Testing flat shape:          {}'.format(self.test_flat.shape))
        result.append(' ')
        result.append('Training label shape:        {}'.format(self.train_labels.shape))
        result.append('Validation flat label shape: {}'.format(self.validate_labels.shape))
        result.append('Testing gray label shape:    {}'.format(self.test_labels.shape))
        result.append(' ')
        result.append('Sign names:')
        result.append(' ')
        for k, v in self.sign_names_map.items():
            result.append('  {} - {}'.format(k, v))
        result.append(' ')
        return '\n'.join(result)


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec


class ImagePlotter:
    @staticmethod
    def plot_image(image):
        # image = mpimg.imread(X_train[0][0])
        # image = X_train[0][0]
        plt.imshow(image, interpolation='nearest')
        # plt.imshow(image)
        plt.axis('off')
        plt.show()

    @staticmethod
    def plot_images1(images, labels, cmap=None):
        gs = gridspec.GridSpec(10, 10)
        gs.update(wspace=0.01, hspace=0.02)  # set the spacing between axes.

        plt.figure(figsize=(12, 12))

        for i in range(len(images)):
            ax = plt.subplot(gs[i])

            ax.set_xticklabels([])
            ax.set_yticklabels([])

            ax.set_aspect('equal')
            xlabel = "T: {0}, P: {1}".format(labels[i], None)
            ax.set_xlabel(xlabel)

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

            plt.subplot(10, 10, i + 1)
            ax.imshow(images[i], cmap=cmap, interpolation='bicubic')
            # plt.axis('off')

        plt.show()

    def plot_images(images, labels, cls_pred=None, cmap=None):
        fig, axes = plt.subplots(4, 5)
        fig.subplots_adjust(hspace=0.5, wspace=0.3)

        for i, ax in enumerate(axes.flat):
            if i >= len(images):
                break
            # Plot image.
            ax.imshow(images[i], cmap=cmap)

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(labels[i])
            else:
                xlabel = "T: {0}, P: {1}".format(labels[i], cls_pred[i])

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()


class ImageTransformer:
    """
    Inspired by several publications by Vivek Yadav one of them from the Udacity Self-Driving Car forums:

    https://carnd-udacity.atlassian.net/wiki/questions/10322627/project-2-unbalanced-data-generating-additional-data-by-jittering-the-original-image
    """

    @staticmethod
    def jitter_images(images):
        # fs for features
        jittered_images = []
        for i in range(len(images)):
            image = images[i]
            jittered_image = ImageTransformer.transform_image(image)
            jittered_images.append(jittered_image)
        return np.array(jittered_images)

    @staticmethod
    def transform_image(image, ang_range=20, shear_range=10, trans_range=5):
        """
        This method was pulled from Udacity Self-Driving Car forums.

        This function transforms images to generate new images.
        The function takes in following arguments,
        1- Image
        2- ang_range: Range of angles for rotation
        3- shear_range: Range of values to apply affine transform to
        4- trans_range: Range of values to apply translations over.

        A Random uniform distribution is used to generate different parameters for transformation
        """

        # Rotation

        ang_rot = np.random.uniform(ang_range) - ang_range / 2
        rows, cols, ch = image.shape
        Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)

        # Translation
        tr_x = trans_range * np.random.uniform() - trans_range / 2
        tr_y = trans_range * np.random.uniform() - trans_range / 2
        Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

        # Shear
        pts1 = np.float32([[5, 5], [20, 5], [5, 20]])

        pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
        pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2

        pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])

        shear_M = cv2.getAffineTransform(pts1, pts2)

        image = cv2.warpAffine(image, Rot_M, (cols, rows))
        image = cv2.warpAffine(image, Trans_M, (cols, rows))
        image = cv2.warpAffine(image, shear_M, (cols, rows))

        return image
