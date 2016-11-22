import hashlib
import os
import pickle
from urllib.request import urlretrieve

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
from tqdm import tqdm
from zipfile import ZipFile
import math

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

training_file = 'traffic-sign-data/train.p'
testing_file = 'traffic-sign-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

train_features, train_labels, train_size, train_coords = train['features'], train['labels'], train['sizes'], train['coords']
test_features, test_labels, test_size, test_coords = test['features'], test['labels'], test['sizes'], test['coords']

assert len(train_features) == len(train_labels), 'features must be same size as labels'

# Reshape train features
train_features = np.arange(len(train_features)*1024).reshape((len(train_features), 1024))
test_features = np.arange(len(test_features)*1024).reshape((len(test_features), 1024))

assert len(train_features) == len(train_labels), 'features must be same size as labels'

_epochs_completed = 0
_index_in_epoch = 0
_num_examples = len(train_features)

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

# Problem 1 - Implement Min-Max scaling for greyscale image data
def normalize_greyscale(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    # ToDo: Implement Min-Max scaling for greyscale image data
    a = 0.1
    b = 0.9
    x_min = np.min(image_data)
    x_max = np.max(image_data)
    x_prime = [a + (((x-x_min)*(b-a))/(x_max-x_min)) for x in image_data]
    # print(image_data, ' normalized to ---> ', x_prime)
    return np.array(x_prime)




# [Adapted from Lesson 7 - MiniFlow]
# Turn labels into numbers and apply One-Hot Encoding
num_channels = 3
train_features = normalize_greyscale(train_features)
test_features = normalize_greyscale(test_features)
num_channels = 1



def compute_dimensions(train_features, test_features):
    n_train = len(train_features)
    n_test = len(test_features)
    image_shape = train_features.shape[1:3]
    labels_count = len(np.unique(train_labels))
    image_size = image_shape[0]

    # Images are stored in one-dimensional arrays of this length.
    img_size_flat = image_size * image_size

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Number of labels =", labels_count)
    print("Image size =", image_size)
    print("img_size_flat =", img_size_flat)
    print("")
    print("")

    return n_train, n_test, num_channels, image_shape, labels_count, image_size, img_size_flat

n_train, n_test, num_channels, image_shape, labels_count, image_size, img_size_flat = compute_dimensions(train_features, test_features)


def next_batch(batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    global _index_in_epoch, _num_examples, _epochs_completed, train_features, train_labels

    start = _index_in_epoch
    _index_in_epoch += batch_size
    if _index_in_epoch > _num_examples:
        # Finished epoch
        _epochs_completed += 1
        # Shuffle the data
        perm = np.arange(_num_examples)
        np.random.shuffle(perm)
        train_features = train_features[perm]
        train_labels = train_labels[perm]
        # Start next epoch
        start = 0
        _index_in_epoch = batch_size
        assert batch_size <= _num_examples
    end = _index_in_epoch
    return train_features[start:end], train_labels[start:end]



# [Adapted from Lesson 7 - MiniFlow]
# Turn labels into numbers and apply One-Hot Encoding
encoder = LabelBinarizer()
encoder.fit(train_labels)
train_labels = encoder.transform(train_labels)
test_labels = encoder.transform(test_labels)

# Change to float32, so it can be multiplied against the features in TensorFlow, which are float32
train_labels = train_labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)
is_labels_encod = True

print('Labels One-Hot Encoded')



# Get randomized datasets for training and validation
print('train_features before split: ', len(train_features))
print('train_labels before split: ', len(train_labels))
print('test_features before split: ', len(test_features))
print('test_labels before split: ', len(test_labels))
train_features, valid_features, train_labels, valid_labels = train_test_split(
    train_features,
    train_labels,
    test_size=0.05,
    random_state=832289)

print('Training features and labels randomized and split.')

print('train_features after split: ', len(train_features))
print('train_labels after split: ', len(train_labels))
print('test_features after split: ', len(test_features))
print('test_labels after split: ', len(test_labels))


features_count = image_size*num_channels  # TRAFFIC SIGNS data input (img shape: 32*32)

a_mode = 2
b_mode = 1

if a_mode == 1:
    # Parameters
    learning_rate = 0.2
    training_epochs = 5
    batch_size = 100
    display_step = 1

    # labels_count = labels_count  # TRAFFIC SIGNS total classes (0-43 digits)

    n_hidden_layer = 512 # layer number of features

    # Store layers weight & bias
    weights = {
        'hidden_layer': tf.Variable(tf.random_normal([features_count, n_hidden_layer])),
        'out': tf.Variable(tf.random_normal([n_hidden_layer, labels_count]))
    }
    biases = {
        'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
        'out': tf.Variable(tf.random_normal([labels_count]))
    }

    # tf Graph input
    x = tf.placeholder("float", [None, image_shape[0], image_shape[1], num_channels])
    y = tf.placeholder("float", [None, labels_count])

    # x_flat = tf.reshape(x, [-1, features_count])
    x_flat = tf.reshape(x, [-1,image_shape[0], image_shape[1], num_channels])

    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x_flat, weights['hidden_layer']), biases['hidden_layer'])
    layer_1 = tf.nn.relu(layer_1)
    # Output layer with linear activation
    logits = tf.matmul(layer_1, weights['out']) + biases['out']

    if b_mode == 1:
        # Dropout prevention
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(layer_1, keep_prob)

        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y)) # also called cross_entropy
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

        # Initializing the variables
        init = tf.initialize_all_variables()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            # Training cycle
            for epoch in range(training_epochs):
                total_batch = int(n_train/batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_x, batch_y = next_batch(batch_size)
                    # Run optimization op (backprop) and cost op (to get loss value)
                    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                # Display logs per epoch step
                if epoch % display_step == 0:
                    c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                    print("Epoch:", '%04d' % (epoch+1), "cost=", \
                        "{:.9f}".format(c))
            print("Optimization Finished!")

            # Test model
            # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({x: test_features, y: test_labels}))

    elif b_mode == 2:
        # Launch the graph
        with tf.Session() as sess:
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            sess.run(tf.initialize_all_variables())
            for i in range(20000):
                batch = next_batch(batch_size)
                if i % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
                    print("step %d, training accuracy %g" % (i, train_accuracy))
                train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

            print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))

elif a_mode == 2:
    print("DO lesson_7_miniflow lab process")

    # ToDo: Set the features and labels tensors
    features = tf.placeholder(tf.float32)
    labels = tf.placeholder(tf.float32)

    # ToDo: Set the weights and biases tensors
    weights = tf.Variable(tf.truncated_normal([features_count, labels_count]))
    biases = tf.Variable(tf.zeros([labels_count], dtype=tf.float32))

    from tensorflow.python.ops.variables import Variable

    assert features._op.name.startswith('Placeholder'), 'features must be a placeholder'
    assert labels._op.name.startswith('Placeholder'), 'labels must be a placeholder'
    assert isinstance(weights, Variable), 'weights must be a TensorFlow variable'
    assert isinstance(biases, Variable), 'biases must be a TensorFlow variable'

    assert features._shape == None or ( \
        features._shape.dims[0].value is None and \
        features._shape.dims[1].value in [None, 1024]), 'The shape of features is incorrect'
    assert labels._shape in [None, 43], 'The shape of labels is incorrect'
    assert weights._variable._shape == (1024, 43), 'The shape of weights is incorrect'
    assert biases._variable._shape == (43), 'The shape of biases is incorrect'

    assert features._dtype == tf.float32, 'features must be type float32'
    assert labels._dtype == tf.float32, 'labels must be type float32'

    # Feed dicts for training, validation, and test session
    train_feed_dict = {features: train_features, labels: train_labels}
    valid_feed_dict = {features: valid_features, labels: valid_labels}
    test_feed_dict = {features: test_features, labels: test_labels}

    # Linear Function WX + b
    logits = tf.matmul(features, weights) + biases

    prediction = tf.nn.softmax(logits)

    # Cross entropy
    cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), reduction_indices=1)

    # Training loss
    loss = tf.reduce_mean(cross_entropy)

    # Create an operation that initializes all variables
    init = tf.initialize_all_variables()

    # Test Cases
    with tf.Session() as session:
        session.run(init)
        session.run(loss, feed_dict=train_feed_dict)
        session.run(loss, feed_dict=valid_feed_dict)
        session.run(loss, feed_dict=test_feed_dict)
        biases_data = session.run(biases)

    assert not np.count_nonzero(biases_data), 'biases must be zeros'

    print('Tests Passed!')

    # Determine if the predictions are correct
    is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    # Calculate the accuracy of the predictions
    accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

    print('Accuracy function created.')


    # ToDo: Find the best parameters for each configuration
    # Validation accuracy at 0.8085333108901978
    epochs = 5
    batch_size = 50
    learning_rate = 0.2

    ### DON'T MODIFY ANYTHING BELOW ###
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # The accuracy measured against the validation set
    validation_accuracy = 0.0

    # Measurements use for graphing loss and accuracy
    log_batch_step = 50
    batches = []
    loss_batch = []
    train_acc_batch = []
    valid_acc_batch = []

    with tf.Session() as session:
        session.run(init)
        batch_count = int(math.ceil(len(train_features)/batch_size))

        for epoch_i in range(epochs):

            # Progress bar
            batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')

            # The training cycle
            for batch_i in batches_pbar:
                # Get a batch of training features and labels
                batch_start = batch_i*batch_size
                batch_features = train_features[batch_start:batch_start + batch_size]
                batch_labels = train_labels[batch_start:batch_start + batch_size]

                # Run optimizer and get loss
                _, l = session.run(
                    [optimizer, loss],
                    feed_dict={features: batch_features, labels: batch_labels})

                # Log every 50 batches
                if not batch_i % log_batch_step:
                    # Calculate Training and Validation accuracy
                    training_accuracy = session.run(accuracy, feed_dict=train_feed_dict)
                    validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

                    # Log batches
                    previous_batch = batches[-1] if batches else 0
                    batches.append(log_batch_step + previous_batch)
                    loss_batch.append(l)
                    train_acc_batch.append(training_accuracy)
                    valid_acc_batch.append(validation_accuracy)

            # Check accuracy against Validation data
            validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

    loss_plot = plt.subplot(211)
    loss_plot.set_title('Loss')
    loss_plot.plot(batches, loss_batch, 'g')
    loss_plot.set_xlim([batches[0], batches[-1]])
    acc_plot = plt.subplot(212)
    acc_plot.set_title('Accuracy')
    acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
    acc_plot.plot(batches, valid_acc_batch, 'b', label='Validation Accuracy')
    acc_plot.set_ylim([0, 1.0])
    acc_plot.set_xlim([batches[0], batches[-1]])
    acc_plot.legend(loc=4)
    plt.tight_layout()
    plt.show()

    print('Validation accuracy for [{}, {}, {}] at {}'.format(epochs, batch_size, learning_rate, validation_accuracy))



    """
    Test

    Set the epochs, batch_size, and learning_rate with the best learning parameters you discovered in problem 3.  You're
    going to test your model against your hold out dataset/testing data.  This will give you a good indicator of how well
    the model will do in the real world.  You should have a test accuracy of atleast 80%.
    """

    # ToDo: Set the epochs, batch_size, and learning_rate with the best parameters from problem 3
    epochs = 5
    batch_size = 50
    learning_rate = 0.1

    ### DON'T MODIFY ANYTHING BELOW ###
    # The accuracy measured against the test set
    test_accuracy = 0.0

    with tf.Session() as session:
        session.run(init)
        batch_count = int(math.ceil(len(train_features) / batch_size))

        for epoch_i in range(epochs):

            # Progress bar
            batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i + 1, epochs), unit='batches')

            # The training cycle
            for batch_i in batches_pbar:
                # Get a batch of training features and labels
                batch_start = batch_i * batch_size
                batch_features = train_features[batch_start:batch_start + batch_size]
                batch_labels = train_labels[batch_start:batch_start + batch_size]

                # Run optimizer
                _ = session.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})

            # Check accuracy against Test data
            test_accuracy = session.run(accuracy, feed_dict=test_feed_dict)

    assert test_accuracy >= 0.80, 'Test accuracy at {}, should be equal to or greater than 0.80'.format(test_accuracy)
    print('Nice Job! Test Accuracy is {}'.format(test_accuracy))

