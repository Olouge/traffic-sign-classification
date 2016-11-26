import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

# TODO: fill this in based on where you saved the training and testing data
training_file = 'traffic-sign-data/train.p'
testing_file = 'traffic-sign-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train, size_train, coords_train = train['features'], train['labels'], train['sizes'], train['coords']
X_test, y_test, size_test, coords_test = test['features'], test['labels'], test['sizes'], test['coords']

n_train = len(X_train)
n_test = len(X_test)
image_shape = X_train.shape[1:3]
n_classes = len(np.unique(y_train))
image_size = image_shape[0]
num_channels = 3
img_size_flat = np.mul(image_size)

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
print("Image size =", image_size)




# Functions for creating new TensorFlow variables in the given shape and initializing them with random values. Note
# that the initialization is not actually done at this point, it is merely being defined in the TensorFlow graph.

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))




"""
Helper-function for creating a new Convolutional Layer

    This function creates a new convolutional layer in the computational graph for TensorFlow. Nothing is actually calculated here, we are just adding the mathematical formulas to the TensorFlow graph.

    It is assumed that the input is a 4-dim tensor with the following dimensions:

    1. Image number.
    2. Y-axis of each image.
    3. X-axis of each image.
    4. Channels of each image.

    Note that the input channels may either be colour-channels, or it may be filter-channels if the input is produced from a previous convolutional layer.

    The output is another 4-dim tensor with the following dimensions:

    1. Image number, same as input.
    2. Y-axis of each image. If 2x2 pooling is used, then the height and width of the input images is divided by 2.
    3. X-axis of each image. Ditto.
    4. Channels produced by the convolutional filters.
"""

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights



"""
Helper-function for flattening a layer

A convolutional layer produces an output tensor with 4 dimensions. We will add fully-connected layers after the
convolution layers, so we need to reduce the 4-dim tensor to 2-dim which can be used as input to the fully-connected layer.
"""


def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features




"""
Helper-function for creating a new Fully-Connected Layer

This function creates a new fully-connected layer in the computational graph for TensorFlow. Nothing is actually
calculated here, we are just adding the mathematical formulas to the TensorFlow graph.

It is assumed that the input is a 2-dim tensor of shape `[num_images, num_inputs]`. The output is a 2-dim tensor of
shape `[num_images, num_outputs]`.
"""

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer




"""
Placeholder variables

Placeholder variables serve as the input to the TensorFlow computational graph that we may change each time we execute
the graph. We call this feeding the placeholder variables and it is demonstrated further below.

First we define the placeholder variable for the input images. This allows us to change the images that are input to
the TensorFlow graph. This is a so-called tensor, which just means that it is a multi-dimensional vector or matrix.
The data-type is set to `float32` and the shape is set to `[None, img_size_flat]`, where `None` means that the tensor
may hold an arbitrary number of images with each image being a vector of length `img_size_flat`.
"""

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')



"""
The convolutional layers expect `x` to be encoded as a 4-dim tensor so we have to reshape it so its shape is instead
`[num_images, img_height, img_width, num_channels]`. Note that `img_height == img_width == img_size` and `num_images`
can be inferred automatically by using -1 for the size of the first dimension. So the reshape operation is:
"""

x_image = tf.reshape(x, [-1, image_size, image_size, num_channels])



"""
Next we have the placeholder variable for the true labels associated with the images that were input in the placeholder
variable `x`. The shape of this placeholder variable is `[None, num_classes]` which means it may hold an arbitrary
number of labels and each label is a vector of length `num_classes` which is 10 in this case.
"""

y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')

y_true_cls = tf.argmax(y_true, dimension=1)

filter_size1 = 5
filter_size2 = 5

num_filters1 = 16
num_filters2 = 36

layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)



layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)



"""
Flatten Layer

The convolutional layers output 4-dim tensors. We now wish to use these as input in a fully-connected network, which
requires for the tensors to be reshaped or flattened to 2-dim tensors.
"""

layer_flat, num_features = flatten_layer(layer_conv2)

