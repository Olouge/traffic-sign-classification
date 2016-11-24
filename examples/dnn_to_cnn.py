import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

"""
  1. Learning Parameters
"""

# Parameters
learning_rate = 0.001
training_epochs = 25
batch_size = 100
display_step = 1

n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

print('')
print('learning rate:', learning_rate)
print('training epochs:',training_epochs)
print('batch size:',batch_size)
print('display step:',display_step)
print('n inputs:', n_input)
print('n classes:', n_classes)
print('')


# The focus for the class is multilayer neural networks, so you'll be given all the learning parameters.

"""
  2. Hidden Layer Parameters
"""

# The variable n_hidden_layer determines the size of the hidden layer in the neural network. This is also know as
# the width of a layer.
n_hidden_layer = 256  # layer number of features

"""
  3. Input
"""

# tf Graph input
x = tf.placeholder("float", [None, 28, 28, 1])
# Define loss and optimizer
y_ = tf.placeholder("float", [None, n_classes])
x_flat = tf.reshape(x, [-1, n_input])
print('x reshaped from 28*28*1 to {}*1 vector'.format(n_input))

"""
 The MNIST data is made up of 28px by 28px images with a single channel. The tf.reshape function above reshapes a
 batch of 28*28 pixels, x, to a batch of 784 pixels.

 https://en.wikipedia.org/wiki/Channel_(digital_image%29
 https://www.tensorflow.org/api_docs/python/array_ops.html#reshape
"""

"""
  4. Weights and Biases

  Deep neural networks use multiple layers with each layer requiring it's own weight and bias. The 'hidden_layer'
  weight and bias is for the hidden layer. The 'out' weight and bias is for the output layer. If the neural network
  were deeper, there would be weights and biases for each additional layer.
"""

# Store layers weight & bias
weights = {
    'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))
}
biases = {
    'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

"""
 5.  Multilayer Perceptron
"""

# Hidden layer with RELU (Rectified Linear Unit) activation
layer_1 = tf.add(tf.matmul(x_flat, weights['hidden_layer']), biases['hidden_layer'])
layer_1 = tf.nn.relu(layer_1)  # This ensures non-linearity data by turning all negative numbers to zero

# Output layer with linear activation
logits = tf.matmul(layer_1, weights['out']) + biases['out']

"""
You've seen the linear function tf.add(tf.matmul(x_flat, weights['hidden_layer']), biases['hidden_layer']) before, also
known as wx + b. Combining linear functions together using a ReLU will give you a two layer network.
"""

"""
  Optimizer
"""

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_))
cost = tf.reduce_mean(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# NOTE: This is the same optimization technique used in the Intro to TensorFLow lab.

# What TensorFlow actually did in that single line was to add new operations to the computation graph. These
# operations included ones to compute gradients, compute parameter update steps, and apply update steps to the
# parameters.
#
# The returned operation "optimizer", when run, will apply the gradient descent updates to the parameters. Training
# the model can therefore be accomplished by repeatedly running train_step.


# Session
"""
  A TensorFlow Session for use in interactive contexts, such as a shell.

  The only difference with a regular Session is that an InteractiveSession installs itself as the default session on
  construction. The methods Tensor.eval() and Operation.run() will use that session to run ops.

  This is convenient in interactive shells and IPython notebooks, as it avoids having to pass an explicit Session
  object to run ops.

  For example:

    sess = tf.InteractiveSession()
    a = tf.constant(5.0)
    b = tf.constant(6.0)
    c = a * b
    # We can just use 'c.eval()' without passing 'sess'
    print(c.eval())
    sess.close()
"""
# sess = tf.InteractiveSession()

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
# with sess:
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    total_batch = int(mnist.train.num_examples / batch_size)
    for epoch in range(training_epochs):
        # print('[epoch:{}/{}][batch:{}]'.format(epoch, training_epochs, batch_size))
        # Loop over all batches
        for i in range(total_batch):
            # The MNIST library in TensorFlow provides the ability to receive the dataset in batches. Calling
            # the mnist.train.next_batch() function returns a subset of the training data.
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run(optimizer, feed_dict={x: batch_x, y_: batch_y})

    # Test trained model

    """
    Well, first let's figure out where we predicted the correct label. tf.argmax is an extremely useful function which
    gives you the index of the highest entry in a tensor along some axis. For example, tf.argmax(y,1) is the label our
    model thinks is most likely for each input, while tf.argmax(y_,1) is the correct label. We can use tf.equal to check
    if our prediction matches the truth.
    """
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))

    """
    That gives us a list of booleans. To determine what fraction are correct, we cast to floating point numbers and then
    take the mean. For example, [True, False, True, True] would become [1,0,1,1] which would become 0.75.
    """
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    """
    Finally, we ask for our accuracy on our test data.
    """
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))

    # close our interactive TensorFlow session
    # sess.close()
