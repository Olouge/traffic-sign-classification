import tensorflow as tf
import math

from classifiers.base_neural_network import BaseNeuralNetwork


class TemplateNeuralNetwork(BaseNeuralNetwork):
    def fit(self):
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
        self.x = tf.placeholder("float", [None, image_size])
        self.y = tf.placeholder("float", [None, self.data.num_classes])

        self.x_flat = tf.reshape(self.x, [-1, image_size])

        self.keep_prob = tf.placeholder(tf.float32)  # probability to keep units

        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(self.x_flat, weights['hidden_layer']), biases['hidden_layer'])
        layer_1 = tf.nn.relu(layer_1)
        # layer_1 = tf.nn.dropout(layer_1, keep_prob)

        # Output layer with linear activation
        self.logits = tf.matmul(layer_1, weights['out']) + biases['out']

        # Define loss and optimizer
        # cost also called cross_entropy
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.y))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.cost)

        init = tf.initialize_all_variables()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            # Training cycle
            for epoch in range(training_epochs):
                batch_count = int(math.ceil(self.data.num_training / batch_size))
                # Loop over all batches
                for i in range(batch_count):
                    batch_x, batch_y = self.data.next_batch(batch_size)
                    # Run optimization op (backprop) and cost op (to get loss value)
                    sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: 0.5})
                # Display logs per epoch step
                if epoch % display_step == 0:
                    c = sess.run(self.cost, feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: 0.5})
                    print("Epoch:", '%04d' % (epoch + 1), '/', '%04d' % training_epochs, "cost=", "{:.9f}".format(c))

                    # Calculate accuracy
                    correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
                    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

                    print("  learning rate:     ", sess.run(learning_rate))
                    print("  batch size:        ", batch_size)
                    print("  train accuracy:    ",
                          self.accuracy.eval({self.x: batch_x, self.y: batch_y, self.keep_prob: 1.0}))

                    print("  validate accuracy: ",
                          self.accuracy.eval(
                              {self.x: self.data.validate_flat, self.y: self.data.validate_labels,
                               self.keep_prob: 1.0}))
                    print("  test accuracy:     ",
                          self.accuracy.eval(
                              {self.x: self.data.test_flat, self.y: self.data.test_labels, self.keep_prob: 1.0}))
                    print('')

            self.hyper_parameters.learning_rate = sess.run(learning_rate)
            self.cross_entropy = sess.run(self.cost)

            print("Optimization Finished!")

            # Calculate accuracy
            self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))

            print("validate accuracy:",
                  self.accuracy.eval(
                      {self.x: self.data.validate_flat, self.y: self.validate_labels, self.keep_prob: 1.0}))
            print("test accuracy:",
                  self.accuracy.eval({self.x: self.data.test_flat, self.y: self.test_labels, self.keep_prob: 1.0}))

    def __serialize(self, data={}):
        return super(SimpleNeuralNetwork, self).serialize({
                    'cross_entropy': self.cross_entropy,
                    'correct_prediction': self.correct_prediction,
                    'accuracy': self.accuracy
                })
