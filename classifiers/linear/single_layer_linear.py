import tensorflow as tf
import numpy as np
import math
import os

from classifiers.base_neural_network import BaseNeuralNetwork, HyperParametersContext, ConfigurationContext


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

        self.__build_graph()

        features = self.features
        labels = self.labels
        logits = self.logits

        # Feed dicts for training, validation, test and prediction
        train_feed_dict = {features: data.train_flat, labels: data.train_labels}
        valid_feed_dict = {features: data.validate_flat, labels: data.validate_labels}
        test_feed_dict = {features: data.test_flat, labels: data.test_labels}
        predict_feed_dict = {features: data.predict_flat, labels: data.predict_labels}

        # Passing global_step to minimize() will increment it at each step.
        global_step = tf.Variable(0, trainable=False)

        training_epochs = hyper_parameters.epochs
        batch_size = hyper_parameters.batch_size
        num_training = data.num_training
        batch_count = int(math.ceil(num_training / batch_size))
        display_step = 1

        # Define loss and optimizer
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

        if self.config.optimizer_type == ConfigurationContext.OPTIMIZER_TYPE_GRADIENT_DESCENT:
            # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
            learning_rate = tf.train.exponential_decay(learning_rate=hyper_parameters.start_learning_rate,
                                                       global_step=global_step,
                                                       decay_steps=75000, decay_rate=0.96, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss,
                                                                                                global_step=global_step)
        elif self.config.optimizer_type == ConfigurationContext.OPTIMIZER_TYPE_ADAGRAD:
            learning_rate = tf.constant(hyper_parameters.start_learning_rate)
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss)

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

                    # Run optimization op (backprop) and loss op (to get loss value)
                    sess.run(optimizer, feed_dict=batch_feed_dict)

                # Display logs per epoch step and very last batch iteration
                if epoch % display_step == 0 or (epoch == (training_epochs - 1) and i == (batch_count - 1)):
                    total_iterations = (epoch + 1)

                    print("Epoch:", '%04d' % total_iterations, 'of', '%04d' % training_epochs)

                    self.config.hyper_parameters.end_learning_rate = sess.run(learning_rate)
                    self.loss = sess.run(loss, feed_dict=valid_feed_dict)

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

                    print("  loss:              ", "{:.9f}".format(self.loss))
                    print("  batch accuracy:    ", accuracy.eval(batch_feed_dict))
                    print("  train accuracy:    ", accuracy.eval(train_feed_dict))
                    print("  validate accuracy: ", accuracy.eval(valid_feed_dict))
                    print("  test accuracy:     ", accuracy.eval(test_feed_dict))
                    print("  predict accuracy:  ", accuracy.eval(predict_feed_dict))
                    print("  batch size:        ", batch_size)
                    print("  learning rate:     ", sess.run(learning_rate))
                    print('')

                    saved = self.evaluate_accuracy(sess, accuracy.eval(valid_feed_dict), total_iterations)
                    if saved == True:
                        # store the final results for later analysis
                        # self.weights = {
                        #     'hidden_layer': self.weight_variables['hidden_layer'].eval(),
                        #     'out': self.weight_variables['out'].eval()
                        # }
                        # self.biases = {
                        #     'hidden_layer': self.bias_variables['hidden_layer'].eval(),
                        #     'out': self.bias_variables['out'].eval()
                        # }
                        os.system('say "{:.002f}%"'.format(self.validate_accuracy * 100))

                if total_iterations - self.last_improvement > hyper_parameters.required_accuracy_improvement:
                    msg = 'No improvement found in a while, stopping optimization after {} iterations. Final accuracy, {}% at iteration {}.'.format(
                        total_iterations, str(int(self.validate_accuracy * 100)), self.last_improvement)

                    print(msg)
                    os.system('say "{}"'.format(msg))

                    break

            print("Optimization Finished!")

    def top_k(self, images, true_labels, model_name, k=5):
        self.__build_graph()

        features = self.features
        labels = self.labels
        logits = self.logits

        with tf.Session() as sess:
            self.saver = tf.train.Saver()
            self.saver.restore(sess, self.save_dir + '/' + model_name)

            # Calculate predictions.
            # in_top_k_op = tf.nn.in_top_k(logits, true_labels, k)
            top_1_op = tf.nn.top_k(logits, 1)
            top_1 = sess.run(top_1_op, feed_dict={features: images})

            top_k_op = tf.nn.top_k(logits, k)
            top_k = sess.run(top_k_op, feed_dict={features: images})

            y_pred, input = top_1.values, top_1.indices

            print('top 1:')
            print('')
            print(top_1)
            print(top_1.eval())
            print(top_1.eval().shape)

            print('top {}:'.format(k))
            print('')
            print(top_k)
            print(top_k.eval())
            print(top_k.eval().shape)

    def predict(self, images, true_labels, model_name):
        self.__build_graph()

        features = self.features
        labels = self.labels
        logits = self.logits

        predict_feed_dict = {features: images, labels: true_labels}

        with tf.Session() as sess:
            # This seems to take A LOOOOOOONG time so not doing it right now.
            # self.saver = tf.train.import_meta_graph(self.save_dir + '/' + model_name + '.meta')
            # self.saver.restore(sess, self.save_dir + '/' + model_name)
            self.saver = tf.train.Saver()
            self.saver.restore(sess, self.save_dir + '/' + model_name)

            # Calculate accuracy
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            # self.predict_accuracy = accuracy.eval(predict_feed_dict)
            # self.predict_predictions = tf.cast(correct_prediction.eval(predict_feed_dict), "float").eval()

            print("  predict accuracy: {}%".format(math.ceil(accuracy.eval(predict_feed_dict) * 100)))

    def __build_graph(self):
        data = self.config.data
        hyper_parameters = self.config.hyper_parameters

        image_size = data.train_flat.shape[1]
        num_classes = data.num_classes

        n_hidden_layer = hyper_parameters.hidden_layer_neuron_count

        # Store layers weight & bias
        self.weight_variables = {
            'hidden_layer': tf.Variable(tf.random_normal([image_size, n_hidden_layer]), name='weights_hidden_layer'),
            'out': tf.Variable(tf.random_normal([n_hidden_layer, num_classes]), name='weights_out')
        }
        self.bias_variables = {
            # 'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer]), name='biases_hidden_layer'),
            # 'out': tf.Variable(tf.random_normal([num_classes]), name='biases_out')
            'hidden_layer': tf.Variable(tf.zeros([n_hidden_layer]), name='biases_hidden_layer'),
            'out': tf.Variable(tf.zeros([num_classes]), name='biases_out')
        }

        self.features = tf.placeholder("float", [None, image_size])
        self.labels = tf.placeholder("float", [None, num_classes])

        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(self.features, self.weight_variables['hidden_layer']),
                         self.bias_variables['hidden_layer'])
        layer_1 = tf.nn.relu(layer_1)

        # Output layer with linear activation
        self.logits = tf.matmul(layer_1, self.weight_variables['out']) + self.bias_variables['out']
