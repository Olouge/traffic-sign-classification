import tensorflow as tf
import numpy as np
import math

from classifiers.base_neural_network import BaseNeuralNetwork
from datasets.german_traffic_signs import ImagePlotter, ImageTransformer


class SimpleNeuralNetwork(BaseNeuralNetwork):
    def fit(self):
        data = self.config.data
        hyper_parameters = self.config.hyper_parameters

        image_size = data.train_flat.shape[1]
        num_classes = data.num_classes
        num_training = data.num_training

        # learning_rate = tf.constant(hyper_parameters.learning_rate)

        # Passing global_step to minimize() will increment it at each step.
        global_step = tf.Variable(0, trainable=False)
        initial_learning_rate = hyper_parameters.learning_rate
        learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, 20000, 0.96, staircase=True)

        training_epochs = hyper_parameters.epochs
        batch_size = hyper_parameters.batch_size
        batch_count = int(math.ceil(num_training / batch_size))
        display_step = 1

        n_hidden_layer = 256  # layer number of features

        # Store layers weight & bias
        weights = {
            'hidden_layer': tf.Variable(tf.random_normal([image_size, n_hidden_layer])),
            'out': tf.Variable(tf.random_normal([n_hidden_layer, num_classes]))
        }
        biases = {
            'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
            'out': tf.Variable(tf.random_normal([num_classes]))
        }

        # tf Graph input
        features = tf.placeholder("float", [None, image_size])
        labels = tf.placeholder("float", [None, num_classes])

        # self.x_flat = tf.reshape(features, [-1, image_size])

        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(features, weights['hidden_layer']), biases['hidden_layer'])
        layer_1 = tf.nn.relu(layer_1)

        # Output layer with linear activation
        logits = tf.matmul(layer_1, weights['out']) + biases['out']

        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

        init = tf.initialize_all_variables()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            # Training cycle
            for epoch in range(training_epochs):
                # Loop over all batches
                for i in range(batch_count):
                    batch_x, batch_y, batch_start, batch_end = data.next_batch(batch_size)
                    # ImagePlotter.plot_images(ImageTransformer.jitter_images(data.train_orig[batch_start:batch_end]), batch_y)
                    # ImagePlotter.plot_images(data.train_orig[batch_start:batch_end], np.argmax(batch_y, axis=1))
                    # Run optimization op (backprop) and cost op (to get loss value)
                    sess.run(optimizer, feed_dict={features: batch_x, labels: batch_y})

                # Display logs per epoch step
                if epoch % display_step == 0:
                    c = sess.run(cost, feed_dict={features: batch_x, labels: batch_y})
                    print("Epoch:", '%04d' % (epoch + 1), 'of', '%04d' % training_epochs)

                    # Calculate accuracy
                    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

                    print("  cost:              ", "{:.9f}".format(c))
                    print("  train accuracy:    ", accuracy.eval({features: data.train_flat, labels: data.train_labels}))
                    print("  validate accuracy: ",
                          accuracy.eval(
                              {features: data.validate_flat, labels: data.validate_labels}))
                    print("  test accuracy:     ",
                          accuracy.eval(
                              {features: data.test_flat, labels: data.test_labels}))

                    print("  batch size:        ", batch_size)
                    print("  learning rate:     ", sess.run(learning_rate))
                    print('')

            # store the final results for later analysis
            self.config.hyper_parameters.learning_rate = sess.run(learning_rate)

            self.cost = sess.run(cost, feed_dict={features: data.train_flat, labels: data.train_labels})
            self.weights = {
                'hidden_layer': weights['hidden_layer'].eval(),
                'out': weights['out'].eval()
            }
            self.biases = {
                'hidden_layer': biases['hidden_layer'].eval(),
                'out': biases['out'].eval()
            }

            # Calculate accuracy
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            # store predictions
            self.train_predictions = tf.cast(
                correct_prediction.eval({features: data.train_flat, labels: data.train_labels}), "float").eval()
            self.test_predictions = tf.cast(correct_prediction.eval({features: data.test_flat, labels: data.test_labels}),
                                            "float").eval()
            self.validate_predictions = tf.cast(
                correct_prediction.eval({features: data.validate_flat, labels: data.validate_labels}), "float").eval()

            # store accuracies
            self.train_accuracy = accuracy.eval({features: data.train_flat, labels: data.train_labels})
            self.validate_accuracy = accuracy.eval({features: data.validate_flat, labels: data.validate_labels})
            self.test_accuracy = accuracy.eval({features: data.test_flat, labels: data.test_labels})

            print("cost:                ", self.cost)
            print("train accuracy:      ", self.train_accuracy)
            print("validate accuracy:   ", self.validate_accuracy)
            print("test accuracy:       ", self.test_accuracy)

            print("Optimization Finished!")
