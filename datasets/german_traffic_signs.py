import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os
import csv

from sklearn.preprocessing import LabelBinarizer

from maths.transformations.image_jitterer import ImageJitterer
from plot.image_plotter import ImagePlotter
from serializers.trained_data_serializer import TrainedDataSerializer


class GermanTrafficSignDataset:
    """
    This class contains the following mechanisms:

       1. #configure  - Trains a new network from the original traffic sign dataset
       2. #serialize  - Constructs a dictionary representation of this dataset
       3. #persist    - Saves a training, validation and test set after training a network
       4. #restore    - Restores a serialized training, validation and test set via to feed into another network
       5. Passing an instance of this class to print() prints some hueristics about the dataset.
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

    def configure(self, one_hot=True, train_validate_split_percentage=0.2):
        """
        Pipeline import sequence

          1. Reads in the signnames.csv and constructs a dictionary. The key is the class number and the value is
             the name of the traffic sign.

          2. Loads the original data from the respective pickle files.

          3. Splits the training set into a training and validation set (train_orig and validate_orig, respectively).

          4. Computes the various metrics about the datasets such as the number of train features and the number of
             unique train labels.

          5. Prepares the images by doing the following:

              5a) Places the original images into an "orig" bucket. So the training, validation and test
                 images will be in the train_orig, validate_orig and test_orig buckets respectively.

              5b) Places a grayscale representation of the original images into a "gray" bucket. So the training,
                 validation and test images will be in the train_gray, validate_gray and test_gray buckets,
                 respectively.

              5c) Places a flattened representation of the grayscale images into a "flat" bucket. So the training,
                 validation and test images will be in the train_flat, validate_flat and test_flat buckets,
                 respectively.

              Bucket keys:

                orig:       The original unprocessed images
                gray:       The original unprocessed images with a grayscale filter applied
                flat:       The grayscale images flattened into a vector

          6. If the one_hot parameter is true, then all train, validate and test labels are converted to a
             self.num_classes-demensional vector where each value in the vector is zero except for the index
             corresponding the class number corresponding to the original label value.
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

    def one_hot_encoded(self):
        return self.__one_hot_encoded

    def label_sign_name(self, labels, idx):
        if self.one_hot_encoded:
            label = np.argmax(labels[idx])
        else:
            label = labels[idx]
        return label, self.sign_names_map[label]

    def restore_from_data(self, data):
        """
        Pipeline import sequence

          1. Assigns instance variables based on the attributes from that dictionary.
        """
        self.__from_data(data)
        if not self.__configured:
            [f() for f in [
                self.__compute_metrics
            ]]
            self.__configured = True

    def restore(self, pickle_file='trafficsigns_trained.pickle'):
        """
        Pipeline import sequence

        Loads the original data from the respective pickle files.
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

    def __from_data(self, data):
        if not self.__configured:
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

    def __restore(self, pickle_file='trafficsigns_trained.pickle'):
        if not self.__configured:
            data = TrainedDataSerializer.reload_data(pickle_file=pickle_file)

            self.__from_data(data)

            del data
            print('train features shape: ', self.train_orig.shape)

    def plot_images(self):
        if self.__configured:
            ImagePlotter.plot_images(ImageJitterer.jitter_images(self.train_orig[:20]), self.train_labels[:20])
            ImagePlotter.plot_images(self.train_gray[:20], self.train_labels[:20], cmap='gray')

            ImagePlotter.plot_images(ImageJitterer.jitter_images(self.test_orig[:20]), self.test_labels[:20])
            ImagePlotter.plot_images(self.test_gray[:20], self.test_labels[:20], cmap='gray')

            ImagePlotter.plot_images(ImageJitterer.jitter_images(self.validate_orig[:20]), self.validate_labels[:20])
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
        Prepares the images for training, validation, testing and visualization.

        In particular:

              1. Places the original images into an "orig" bucket. So the training, validation and test
                 images will be in the train_orig, validate_orig and test_orig buckets respectively.

              2. Places a grayscale representation of the original images into a "gray" bucket. So the training,
                 validation and test images will be in the train_gray, validate_gray and test_gray buckets,
                 respectively.

              3. Places a flattened representation of the grayscale images into a "flat" bucket. So the training,
                 validation and test images will be in the train_flat, validate_flat and test_flat buckets,
                 respectively.

        Bucket keys:

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
