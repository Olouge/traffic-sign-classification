import tensorflow as tf
import numpy as np
import math
import os

from classifiers.base_neural_network import BaseNeuralNetwork, HyperParametersContext


class SingleLayerHyperParametersContext(HyperParametersContext):
    def __init__(
            self,
            hidden_layer_neuron_count=512,
            **kwargs
    ):
        """

        :param hidden_layer_neuron_count: number of neurons for the hidden layer
        :param kwargs: Arguments to pass into to super constructor
        """
        super(SingleLayerHyperParametersContext, self).__init__(**kwargs)
        self.hidden_layer_neuron_count = hidden_layer_neuron_count


class SingleLayerLinear(BaseNeuralNetwork):
    def fit(self):
        data = self.config.data
        hyper_parameters = self.config.hyper_parameters

        image_size = data.train_flat.shape[1]
        num_classes = data.num_classes
        num_training = data.num_training

        # learning_rate = tf.constant(hyper_parameters.start_learning_rate)

        # Passing global_step to minimize() will increment it at each step.
        global_step = tf.Variable(0, trainable=False)
        initial_learning_rate = hyper_parameters.start_learning_rate

        # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        decayed_learning_rate = tf.train.exponential_decay(learning_rate=initial_learning_rate, global_step=global_step,
                                                           decay_steps=75000, decay_rate=0.96, staircase=True)

        training_epochs = hyper_parameters.epochs
        batch_size = hyper_parameters.batch_size
        batch_count = int(math.ceil(num_training / batch_size))
        display_step = 1

        n_hidden_layer = hyper_parameters.hidden_layer_neuron_count

        # Store layers weight & bias
        weights = {
            'hidden_layer': tf.Variable(tf.random_normal([image_size, n_hidden_layer]), name='weights_hidden_layer'),
            'out': tf.Variable(tf.random_normal([n_hidden_layer, num_classes]), name='weights_out')
        }
        biases = {
            'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer]), name='biases_hidden_layer'),
            'out': tf.Variable(tf.random_normal([num_classes]), name='biases_out')
        }

        # tf Graph input
        features = tf.placeholder("float", [None, image_size])
        labels = tf.placeholder("float", [None, num_classes])

        # Feed dicts for training, validation, and test session
        train_feed_dict = {features: data.train_flat, labels: data.train_labels}
        valid_feed_dict = {features: data.validate_flat, labels: data.validate_labels}
        test_feed_dict = {features: data.test_flat, labels: data.test_labels}
        predict_feed_dict = {features: data.predict_flat, labels: data.predict_labels}

        # x_flat = tf.reshape(features, [-1, image_size])

        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(features, weights['hidden_layer']), biases['hidden_layer'])
        layer_1 = tf.nn.relu(layer_1)

        # Output layer with linear activation
        logits = tf.matmul(layer_1, weights['out']) + biases['out']

        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=decayed_learning_rate).minimize(cost,
                                                                                                    global_step=global_step)

        init = tf.initialize_all_variables()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            # Training cycle
            for epoch in range(training_epochs):
                # Loop over all batches
                for i in range(batch_count):
                    x_batch, y_batch, batch_start, batch_end = data.next_batch(batch_size)
                    batch_feed_dict = {features: x_batch, labels: y_batch}

                    # ImagePlotter.plot_images(ImageJitterer.jitter_images(data.train_orig[batch_start:batch_end]), batch_y)
                    # ImagePlotter.plot_images(data.train_orig[batch_start:batch_end], np.argmax(batch_y, axis=1))

                    # Run optimization op (backprop) and cost op (to get loss value)
                    sess.run(optimizer, feed_dict=batch_feed_dict)

                # Display logs per epoch step and very last batch iteration
                if epoch % display_step == 0 or (epoch == (training_epochs - 1) and i == (batch_count - 1)):
                    total_iterations = (epoch + 1)

                    print("Epoch:", '%04d' % total_iterations, 'of', '%04d' % training_epochs)

                    self.config.hyper_parameters.end_learning_rate = sess.run(decayed_learning_rate)
                    self.cost = sess.run(cost, feed_dict=valid_feed_dict)

                    # Calculate accuracy
                    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

                    # store accuracies
                    self.train_accuracy = accuracy.eval(train_feed_dict)
                    self.validate_accuracy = accuracy.eval(valid_feed_dict)
                    self.test_accuracy = accuracy.eval(test_feed_dict)
                    self.predict_accuracy = accuracy.eval(predict_feed_dict)

                    # store predictions
                    self.train_predictions = tf.cast(correct_prediction.eval(train_feed_dict), "float").eval()
                    self.test_predictions = tf.cast(correct_prediction.eval(test_feed_dict), "float").eval()
                    self.predict_predictions = tf.cast(correct_prediction.eval(predict_feed_dict), "float").eval()
                    self.validate_predictions = tf.cast(correct_prediction.eval(valid_feed_dict), "float").eval()

                    print("  cost:              ", "{:.9f}".format(self.cost))
                    print("  batch accuracy:    ", accuracy.eval(batch_feed_dict))
                    print("  train accuracy:    ", accuracy.eval(train_feed_dict))
                    print("  validate accuracy: ", accuracy.eval(valid_feed_dict))
                    print("  test accuracy:     ", accuracy.eval(test_feed_dict))
                    print("  predict accuracy:  ", accuracy.eval(predict_feed_dict))
                    print("  batch size:        ", batch_size)
                    print("  learning rate:     ", sess.run(decayed_learning_rate))
                    print('')

                    saved = self.evaluate_accuracy(sess, accuracy.eval(valid_feed_dict), total_iterations)
                    if saved == True:
                        # store the final results for later analysis
                        self.weights = {
                            'hidden_layer': weights['hidden_layer'].eval(),
                            'out': weights['out'].eval()
                        }
                        self.biases = {
                            'hidden_layer': biases['hidden_layer'].eval(),
                            'out': biases['out'].eval()
                        }
                        os.system('say "{:.002f}%"'.format(self.validate_accuracy * 100))

                if total_iterations - self.last_improvement > hyper_parameters.required_accuracy_improvement:
                    msg = 'No improvement found in a while, stopping optimization after {} iterations. Final accuracy, {}% at iteration {}.'.format(
                        total_iterations, str(int(self.validate_accuracy * 100)), self.last_improvement)

                    print(msg)
                    os.system('say "{}"'.format(msg))

                    # Break out from the for-loop.
                    break

            print("Optimization Finished!")

    def predict(self, model_name):
        data = self.config.data
        hyper_parameters = self.config.hyper_parameters

        image_size = data.train_flat.shape[1]
        num_classes = data.num_classes

        # Passing global_step to minimize() will increment it at each step.
        global_step = tf.Variable(0, trainable=False)

        n_hidden_layer = hyper_parameters.hidden_layer_neuron_count

        # Store layers weight & bias
        weights = {
            'hidden_layer': tf.Variable(tf.random_normal([image_size, n_hidden_layer]), name='weights_hidden_layer'),
            'out': tf.Variable(tf.random_normal([n_hidden_layer, num_classes]), name='weights_out')
        }
        biases = {
            'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer]), name='biases_hidden_layer'),
            'out': tf.Variable(tf.random_normal([num_classes]), name='biases_out')
        }

        features = tf.placeholder("float", [None, image_size])
        labels = tf.placeholder("float", [None, num_classes])

        predict_feed_dict = {features: data.predict_flat, labels: data.predict_labels}

        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(features, weights['hidden_layer']), biases['hidden_layer'])
        layer_1 = tf.nn.relu(layer_1)

        # Output layer with linear activation
        logits = tf.matmul(layer_1, weights['out']) + biases['out']

        with tf.Session() as sess:
            self.saver = tf.train.Saver()
            self.saver.restore(sess, self.save_dir + '/' + model_name)

            # Calculate accuracy
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            self.predict_accuracy = accuracy.eval(predict_feed_dict)
            self.predict_predictions = tf.cast(correct_prediction.eval(predict_feed_dict), "float").eval()
            print("  predict accuracy:  ", accuracy.eval(predict_feed_dict))

