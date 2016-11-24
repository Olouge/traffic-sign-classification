import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

from pipelines.german_traffic_sign_dataset import GermanTrafficSignDataset

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





data = GermanTrafficSignDataset()
print(data)

# X_train, train_labels, X_train_orig, X_train_gray = data.train_flat, data.train_labels, data.train_orig, data.train_gray
# X_validate, validate_labels, X_validate_orig, X_validate_gray = data.validate_flat, data.validate_labels, data.validate_orig, data.validate_gray
# X_test, test_labels, X_test_orig, X_test_gray = data.test, data.test_labels, data.test_orig, data.test_gray

assert len(data.train_flat) == len(data.train_labels), 'features must be same size as labels'

# Detect if any images' sizes differ from their coords ROI
# for i in range(len(train_coords)):
#     if not np.array_equal(train_size[i], train_coords[i]):
#         print("size: {}         coords: {}".format(train_size[i], train_coords[i]))



image_size = data.train_flat.shape[1]
print(image_size)

# [Adapted from Lesson 7 - MiniFlow]
# Turn labels into numbers and apply One-Hot Encoding
encoder = LabelBinarizer()
encoder.fit(data.train_labels)

train_labels = encoder.transform(data.train_labels)
validate_labels = encoder.transform(data.validate_labels)
test_labels = encoder.transform(data.test_labels)

# Change to float32, so it can be multiplied against the features in TensorFlow, which are float32
train_labels = train_labels.astype(np.float32)
validate_labels = validate_labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)
is_labels_encod = True

print('Labels One-Hot Encoded')


a_mode = 1
b_mode = 1

if a_mode == 1:
    # Parameters
    global_step = tf.Variable(0, trainable=False)
    # learning_rate = tf.constant(0.2)
    initial_learning_rate = 0.5
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, 10000, 0.96, staircase=True)
    # Passing global_step to minimize() will increment it at each step.

    training_epochs = 500
    batch_size = 20
    display_step = 1

    n_hidden_layer = 256  # layer number of features
    n2_hidden_layer = 512  # layer number of features

    # Store layers weight & bias
    weights = [
        {
            'hidden_layer': tf.Variable(tf.random_normal([image_size, n_hidden_layer])),
            'out': tf.Variable(tf.random_normal([n_hidden_layer, data.num_classes]))
        },
        {
            'hidden_layer': tf.Variable(tf.random_normal([image_size, n2_hidden_layer])),
            'out': tf.Variable(tf.random_normal([n2_hidden_layer, data.num_classes]))
        }
    ]
    biases = [
        {
            'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
            'out': tf.Variable(tf.random_normal([data.num_classes]))
        },
        {
            'hidden_layer': tf.Variable(tf.random_normal([n2_hidden_layer])),
            'out': tf.Variable(tf.random_normal([data.num_classes]))
        }
    ]

    # tf Graph input
    x = tf.placeholder("float", [None, image_size])
    y = tf.placeholder("float", [None, data.num_classes])

    x_flat = tf.reshape(x, [-1, image_size])

    keep_prob = tf.placeholder(tf.float32)  # probability to keep units

    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x_flat, weights[0]['hidden_layer']), biases[0]['hidden_layer'])
    layer_1 = tf.nn.relu(layer_1)
    # layer_1 = tf.nn.dropout(layer_1, keep_prob)

    # Output layer with linear activation
    logits = tf.matmul(layer_1, weights[0]['out']) + biases[0]['out']

    if b_mode == 1:
        # Define loss and optimizer
        # cost also called cross_entropy
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

        init = tf.initialize_all_variables()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            # Training cycle
            for epoch in range(training_epochs):
                total_batch = int(data.num_training / batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_x, batch_y = data.next_batch(batch_size)
                    # Run optimization op (backprop) and cost op (to get loss value)
                    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
                # Display logs per epoch step
                if epoch % display_step == 0:
                    c = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
                    print("Epoch:", '%04d' % (epoch + 1), '/', '%04d'%(training_epochs), "cost=", "{:.9f}".format(c))

                    # Calculate accuracy
                    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    print("  learning rate:     ", sess.run(learning_rate))
                    print("  batch size:        ", batch_size)
                    print("  train accuracy:    ", accuracy.eval({x: batch_x, y: batch_y, keep_prob: 1.0}))
                    print("  test accuracy:     ", accuracy.eval({x: data.test, y: test_labels, keep_prob: 1.0}))
                    print('')
            print("Optimization Finished!")

            # Calculate accuracy
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("test accuracy:", accuracy.eval({x: data.test, y: test_labels, keep_prob: 1.0}))

    elif b_mode == 2:
        # Launch the graph
        with tf.Session() as sess:
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            sess.run(tf.initialize_all_variables())
            for i in range(20000):
                batch = data.next_batch(batch_size)
                if i % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
                    print("step %d, training accuracy %g" % (i, train_accuracy))
                train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

            print("test accuracy %g" % accuracy.eval(feed_dict={x: data.test, y: test_labels, keep_prob: 1.0}))

elif a_mode == 2:
    print("DO lesson_7_miniflow lab process")

    # ToDo: Set the features and labels tensors
    features = tf.placeholder(tf.float32)
    labels = tf.placeholder(tf.float32)

    # ToDo: Set the weights and biases tensors
    weights = tf.Variable(tf.truncated_normal([data.train_flat.shape, data.num_classes]))
    biases = tf.Variable(tf.zeros([data.num_classes], dtype=tf.float32))

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
    train_feed_dict = {features: data.train, labels: train_labels}
    valid_feed_dict = {features: validate_features, labels: validate_labels}
    test_feed_dict = {features: data.test, labels: test_labels}

    # Linear Function WX + b
    # logits = tf.matmul(features, weights) + biases
    x_flat = tf.reshape(features, [-1, data.train_flat.shape])
    logits = tf.matmul(x_flat, weights) + biases

    # From Sridhar Sampath in forums at https://carnd-udacity.atlassian.net/wiki/questions/12617346/answers/12620228
    logits = -np.amax(logits)

    prediction = tf.nn.softmax(logits)

    # Cross entropy
    # cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), reduction_indices=1)
    # From Vivek forum tip at:
    #
    # https://carnd-udacity.atlassian.net/wiki/cq/viewquestion.action?id=12617346&questionTitle=what-could-be-causing-very-low-accuracy
    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(labels * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)), reduction_indices=[1]))

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
    learning_rate = 0.0001
    epochs = 15
    batch_size = 25

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
        batch_count = int(math.ceil(len(data.train) / batch_size))

        for epoch_i in range(epochs):

            # Progress bar
            batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i + 1, epochs), unit='batches')

            # The training cycle
            for batch_i in batches_pbar:
                # Get a batch of training features and labels
                batch_start = batch_i * batch_size
                batch_features = data.train[batch_start:batch_start + batch_size]
                batch_labels = data.train_labels[batch_start:batch_start + batch_size]

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
            print('Epoch {}, validation accuracy {}'.format(epoch_i, validation_accuracy))

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
    epochs = 100
    batch_size = 20
    learning_rate = 0.5

    ### DON'T MODIFY ANYTHING BELOW ###
    # The accuracy measured against the test set
    test_accuracy = 0.0

    with tf.Session() as session:
        session.run(init)
        batch_count = int(math.ceil(len(data.train) / batch_size))

        for epoch_i in range(epochs):

            # Progress bar
            batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i + 1, epochs), unit='batches')

            # The training cycle
            for batch_i in batches_pbar:
                # Get a batch of training features and labels
                batch_start = batch_i * batch_size
                batch_features = data.train[batch_start:batch_start + batch_size]
                batch_labels = train_labels[batch_start:batch_start + batch_size]

                # Run optimizer
                _ = session.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})

            # Check accuracy against Test data
            test_accuracy = session.run(accuracy, feed_dict=test_feed_dict)

    assert test_accuracy >= 0.80, 'Test accuracy at {}, should be equal to or greater than 0.80'.format(test_accuracy)
    print('Nice Job! Test Accuracy is {}'.format(test_accuracy))
