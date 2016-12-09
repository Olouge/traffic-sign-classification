import cv2
import numpy as np


class ImageJitterer:
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
            jittered_image = ImageJitterer.transform_image(image)
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
