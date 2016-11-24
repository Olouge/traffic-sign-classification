import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os


class GermanTrafficSignDataset:
    def __init__(self, train_validate_split_percentage=0.2):
        self.split_size = train_validate_split_percentage

        self.train_orig, self.validate_orig, self.test_orig = None, None, None
        self.train_gray, self.validate_gray, self.test_gray = None, None, None
        self.train_flat, self.validate_flat, self.test_flat = None, None, None
        self.train_resized, self.validate_resized, self.test_resized = None, None, None

        self.train_labels, self.train_size, self.train_coords = None, None, None
        self.validate_labels, self.validate_size, self.validate_coords = None, None, None
        self.test_labels, self.test_size, self.test_coords = None, None, None

        self.num_training, self.num_validation, self.num_testing, self.num_classes = None, None, None, None

        # batch training metrics
        self._epochs_completed = 0
        self._index_in_epoch = 0

        # Pipeline import sequence
        [f() for f in [
            self.load_data,
            self.split_train_and_validation,
            self.prepare_images
        ]]

    def __str__(self):
        result = []
        result.append(' ')
        result.append('Training size:               {}'.format(self.num_training))
        result.append('Validation size:             {}'.format(self.num_validation))
        result.append('Testing size:                {}'.format(self.num_testing))
        result.append('Total classes:               {}'.format(self.num_classes))
        result.append(' ')
        result.append('Training orig shape:         {}'.format(self.train_orig.shape))
        result.append('Training gray shape:         {}'.format(self.train_gray.shape))
        result.append('Training resized shape:      {}'.format(self.train_resized.shape))
        result.append('Training flat shape:         {}'.format(self.train_flat.shape))
        result.append(' ')
        result.append('Validation orig shape:       {}'.format(self.validate_orig.shape))
        result.append('Validation gray shape:       {}'.format(self.validate_gray.shape))
        result.append('Validation resized shape:    {}'.format(self.validate_resized.shape))
        result.append('Validation flat shape:       {}'.format(self.validate_flat.shape))
        result.append(' ')
        result.append('Testing orig shape:          {}'.format(self.test_orig.shape))
        result.append('Testing gray shape:          {}'.format(self.test_gray.shape))
        result.append('Testing resized shape:       {}'.format(self.test_resized.shape))
        result.append('Testing flat shape:          {}'.format(self.test_flat.shape))
        result.append(' ')
        result.append('Training label shape:        {}'.format(self.train_labels.shape))
        result.append('Validation flat label shape: {}'.format(self.validate_labels.shape))
        result.append('Testing gray label shape:    {}'.format(self.test_labels.shape))
        return '\n'.join(result)

    def load_data(self):
        """
        Loads in train and test pickle files
        """

        training_file = os.path.join(os.path.dirname(__file__), 'traffic-sign-data/train.p')
        testing_file = os.path.join(os.path.dirname(__file__), 'traffic-sign-data/test.p')

        with open(training_file, mode='rb') as f:
            train = pickle.load(f)
        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)

        self.train_orig, self.train_labels, self.train_size, self.train_coords = train['features'], train['labels'], \
                                                                                 train['sizes'], train['coords']

        self.test_orig, self.test_labels, self.test_size, self.test_coords = test['features'], test['labels'], \
                                                                             test['sizes'], test['coords']

    def split_train_and_validation(self):
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

        self.num_training = len(self.train_orig)
        self.num_validation = len(self.validate_orig)
        self.num_testing = len(self.test_orig)
        self.num_classes = len(np.unique(self.train_labels))

    def prepare_images(self):
        """
        This method bucketizes the original images.

            orig:       The original unprocessed images
            gray:       The original unprocessed images with a grayscale filter applied
            resized:    The grayscale images resized to the 32 x 32 based on their `sizes` region of interest
            flat:       The grayscale images flattened into a vector
        """
        train_orig_images = []
        train_gray_images = []
        train_flat_images = []
        train_resized_images = []

        validate_orig_images = []
        validate_gray_images = []
        validate_flat_images = []
        validate_resized_images = []

        test_orig_images = []
        test_gray_images = []
        test_flat_images = []
        test_resized_images = []

        for image in self.train_orig:
            gray_image = self.color2gray(image)
            # TODO Look at the sizes/coordinates for this image to resize it
            resized_image = gray_image
            flat_image = np.array(resized_image, dtype=np.float32).flatten()

            train_orig_images.append(image)
            train_gray_images.append(gray_image)
            train_resized_images.append(resized_image)
            train_flat_images.append(flat_image)

        for image in self.validate_orig:
            gray_image = self.color2gray(image)
            # TODO Look at the sizes/coordinates for this image to resize it
            resized_image = gray_image
            flat_image = np.array(resized_image, dtype=np.float32).flatten()

            validate_orig_images.append(image)
            validate_gray_images.append(gray_image)
            validate_resized_images.append(resized_image)
            validate_flat_images.append(flat_image)

        for image in self.test_orig:
            gray_image = self.color2gray(image)
            # TODO Look at the sizes/coordinates for this image to resize it
            resized_image = gray_image
            flat_image = np.array(resized_image, dtype=np.float32).flatten()

            test_orig_images.append(image)
            test_gray_images.append(gray_image)
            test_resized_images.append(resized_image)
            test_flat_images.append(flat_image)

        # orig bucket
        self.train_orig = np.array(train_orig_images)
        self.validate_orig = np.array(validate_orig_images)
        self.test_orig = np.array(test_orig_images)

        # gray bucket
        self.train_gray = np.array(train_gray_images)
        self.validate_gray = np.array(validate_gray_images)
        self.test_gray = np.array(test_gray_images)

        # resized bucket
        self.train_resized = np.array(train_resized_images)
        self.validate_resized = np.array(validate_resized_images)
        self.test_resized = np.array(test_resized_images)

        # flat bucket
        self.train_flat = self.normalize_greyscale(train_flat_images)
        self.validate_flat = self.normalize_greyscale(validate_flat_images)
        self.test_flat = self.normalize_greyscale(test_flat_images)

    def color2gray(self, image):
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
        return gray

    def normalize_greyscale(self, image_data):
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
        return self.train_flat[start:end], self.train_labels[start:end]

    def persist(self, pickle_file='trafficsigns_trained.pickle', overwrite=False):
        TrainedDataMarshaler.save_data(pickle_file=pickle_file, overwrite=overwrite, train_features=self.train_orig,
                                       train_labels=self.train_labels,
                                       validate_features=self.validate_orig, validate_labels=self.validate_labels,
                                       test_features=self.test_orig, test_labels=self.test_labels)

    def restore(self, pickle_file='trafficsigns_trained.pickle'):
        train_features, train_labels, validate_features, validate_labels, test_features, test_labels = TrainedDataMarshaler.reload_data(
            pickle_file=pickle_file)
        print(train_features.shape)


class TrainedDataMarshaler:
    # Save the data for easy access
    @staticmethod
    def save_data(
            train_features,
            train_labels,

            validate_features,
            validate_labels,

            test_features,
            test_labels,

            pickle_file='trafficsigns_trained.pickle',
            overwrite=False
    ):
        pickle_file = os.path.join(os.path.dirname(__file__), 'data', pickle_file)
        os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
        if overwrite or not os.path.isfile(pickle_file):
            print('Saving data to pickle file...')
            try:
                with open(pickle_file, 'wb') as pfile:
                    pickle.dump(
                        {
                            'train_dataset': train_features,
                            'train_labels': train_labels,
                            'validate_dataset': validate_features,
                            'validate_labels': validate_labels,
                            'test_dataset': test_features,
                            'test_labels': test_labels
                        },
                        pfile, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', pickle_file, ':', e)
                raise
        else:
            print('WARNING: {} already exists.'.format(pickle_file))

        print('Data cached in pickle file.')

    @staticmethod
    def reload_data(pickle_file='trafficsigns_trained.pickle'):
        pickle_file = os.path.join(os.path.dirname(__file__), 'data', pickle_file)
        os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
        # Reload the data
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
            train_features = pickle_data['train_dataset']
            train_labels = pickle_data['train_labels']
            validate_features = pickle_data['validate_dataset']
            validate_labels = pickle_data['validate_labels']
            test_features = pickle_data['test_dataset']
            test_labels = pickle_data['test_labels']
            del pickle_data  # Free up memory

        print('Data and modules loaded.')

        return train_features, train_labels, validate_features, validate_labels, test_features, test_labels


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec


class ImagePlotter:
    @staticmethod
    def plot_image(image):
        # image = mpimg.imread(X_train[0][0])
        # image = X_train[0][0]
        plt.imshow(image, interpolation='nearest')
        plt.axis('off')
        plt.show()

    @staticmethod
    def plot_images(images):
        gs1 = gridspec.GridSpec(10, 10)
        gs1.update(wspace=0.01, hspace=0.02)  # set the spacing between axes.
        plt.figure(figsize=(12, 12))
        for i in range(len(images)):
            ax1 = plt.subplot(gs1[i])
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect('equal')

            plt.subplot(10, 10, i + 1)
            plt.imshow(images[i], interpolation='nearest')
            plt.axis('off')
        plt.show()


class ImageTransformer:
    """
    Inspired by several publications by Vivek Yadav one of them from the Udacity Self-Driving Car forums:

    https://carnd-udacity.atlassian.net/wiki/questions/10322627/project-2-unbalanced-data-generating-additional-data-by-jittering-the-original-image
    """

    @staticmethod
    def jitter_images(self, images):
        # fs for features
        jittered_images = []
        for i in range(len(images)):
            image = images[i]
            jittered_image = self.transform_image(image)
            jittered_images.append(jittered_image)
        return np.array(jittered_images)

    @staticmethod
    def transform_image(self, image, ang_range=20, shear_range=10, trans_range=5):
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