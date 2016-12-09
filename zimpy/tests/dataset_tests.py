from zimpy.datasets.german_traffic_signs import GermanTrafficSignDataset

data = GermanTrafficSignDataset()
data.configure(one_hot=False)

X_train = data.train_orig
y_train = data.train_labels

import unittest
import numpy as np
from sklearn.preprocessing import normalize


# Learn more about unit tests here at https://docs.python.org/2/library/unittest.html
class TestDataset(unittest.TestCase):
    def test_shapes(self):
        self.assertEqual(X_train.shape[0], y_train.shape[0],
                         "The number of images is not equal to the number of labels.")
        self.assertEqual(X_train.shape[1:], (32, 32, 3), "The dimensions of the images are not 32 x 32 x 3.")

    def test_data_normalization(self):
        X_norm = data.normalize_data(X_train)

        self.assertEqual(round(np.mean(X_norm)), 0, "The mean of the input data is: %f" % np.mean(X_norm))
        self.assertTupleEqual((np.min(X_norm), np.max(X_norm)), (-0.5, 0.5)), "The range of the input data is: %.1f to %.1f" % (np.min(X_norm), np.max(X_norm))


if __name__ == '__main__':
    unittest.main()
