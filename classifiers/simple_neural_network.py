import tensorflow as tf
import numpy as np
import math

from classifiers.base_neural_network import BaseNeuralNetwork
from datasets.german_traffic_sign_dataset import ImagePlotter, ImageTransformer


class SimpleNeuralNetwork(BaseNeuralNetwork):
    def fit(self):
        print(self)
        print('')

        learning_rate = tf.constant(self.hyper_parameters.learning_rate)

        image_size = self.data.train_flat.shape[1]
        training_epochs = self.hyper_parameters.epochs
        batch_size = self.hyper_parameters.batch_size
        display_step = 1

        n_hidden_layer = 256  # layer number of features

        # Store layers weight & bias
        weights = {
            'hidden_layer': tf.Variable(tf.random_normal([image_size, n_hidden_layer])),
            'out': tf.Variable(tf.random_normal([n_hidden_layer, self.data.num_classes]))
        }
        biases = {
            'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
            'out': tf.Variable(tf.random_normal([self.data.num_classes]))
        }

        # tf Graph input
        x = tf.placeholder("float", [None, image_size])
        y = tf.placeholder("float", [None, self.data.num_classes])

        # self.x_flat = tf.reshape(x, [-1, image_size])

        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['hidden_layer']), biases['hidden_layer'])
        layer_1 = tf.nn.relu(layer_1)

        # Output layer with linear activation
        logits = tf.matmul(layer_1, weights['out']) + biases['out']

        # Define loss and optimizer
        # cost also called cross_entropy
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        init = tf.initialize_all_variables()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            # Training cycle
            for epoch in range(training_epochs):
                batch_count = int(math.ceil(self.data.num_training / batch_size))
                # Loop over all batches
                for i in range(batch_count):
                    batch_x, batch_y, batch_start, batch_end = self.data.next_batch(batch_size)
                    # ImagePlotter.plot_images(ImageTransformer.jitter_images(self.data.train_orig[batch_start:batch_end]), batch_y)
                    ImagePlotter.plot_images(self.data.train_orig[batch_start:batch_end], np.argmax(batch_y, axis=1))
                    # Run optimization op (backprop) and cost op (to get loss value)
                    sess.run(self.optimizer, feed_dict={x: batch_x, y: batch_y})
                # Display logs per epoch step
                if epoch % display_step == 0:
                    c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                    print("Epoch:", '%04d' % (epoch + 1), '/', '%04d' % training_epochs, "cost=", "{:.9f}".format(c))

                    # Calculate accuracy
                    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
                    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

                    print("  learning rate:     ", sess.run(learning_rate))
                    print("  batch size:        ", batch_size)
                    print("  train accuracy:    ",
                          self.accuracy.eval({x: batch_x, y: batch_y}))

                    print("  validate accuracy: ",
                          self.accuracy.eval(
                              {x: self.data.validate_flat, y: self.data.validate_labels}))
                    print("  test accuracy:     ",
                          self.accuracy.eval(
                              {x: self.data.test_flat, y: self.data.test_labels}))
                    print('')

            self.hyper_parameters.learning_rate = sess.run(learning_rate)
            self.cross_entropy = c
            self.weights = {
                'hidden_layer': weights['hidden_layer'].eval(),
                'out': weights['out'].eval()
            }
            self.biases = {
                'hidden_layer': biases['hidden_layer'].eval(),
                'out': biases['out'].eval()
            }

            # Calculate accuracy
            self.correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))

            self.train_accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
            self.validate_accuracy = self.accuracy.eval({x: self.data.validate_flat, y: self.data.validate_labels})
            self.test_accuracy = self.accuracy.eval({x: self.data.test_flat, y: self.data.test_labels})

            print("train accuracy:      ", self.train_accuracy)
            print("validate accuracy:   ", self.validate_accuracy)
            print("test accuracy:       ", self.test_accuracy)

            print("Optimization Finished!")

    def __serialize(self, data={}):
        return super(SimpleNeuralNetwork, self).__serialize({
            'cross_entropy': self.cross_entropy,
            'correct_prediction': self.correct_prediction,
            'train_accuracy': self.train_accuracy,
            'validate_accuracy': self.validate_accuracy,
            'test_accuracy': self.test_accuracy,
            'weights': self.weights,
            'biases': self.biases
        })
