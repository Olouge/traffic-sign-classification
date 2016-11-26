from datetime import datetime

import os
import uuid

import tensorflow as tf
import numpy as np

from serializers.trained_data_serializer import TrainedDataSerializer


class HyperParametersContext:
    def __init__(
            self,
            start_learning_rate=0.2,
            end_learning_rate=0.2,
            epochs=15,
            batch_size=20,
            exponential_decay_config=None,
            required_accuracy_improvement=100
    ):
        self.start_learning_rate = start_learning_rate
        self.end_learning_rate = end_learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.exponential_decay_config = exponential_decay_config

        # Stop optimization if no improvement found in this many iterations.
        self.required_accuracy_improvement = required_accuracy_improvement

    def __str__(self):
        result = []
        result.append(' ')
        for k, v in self.__dict__.items():
            result.append('  {} - {}'.format(k, v))
        result.append(' ')
        return '\n'.join(result)


class ConfigurationContext:
    def __init__(self, dataset, hyper_parameters=None):
        """
        :param dataset: An instance of datasets.GermanTrafficSignDataset
        :param hyper_parameters: An instance of HyperParametersContext
        """
        self.data = dataset
        if hyper_parameters is None:
            self.hyper_parameters = HyperParametersContext()
        else:
            self.hyper_parameters = hyper_parameters


class BaseNeuralNetwork:
    def __init__(self):
        self.uuid = uuid.uuid4()

        self.config = None

        self.cost = None
        self.weights = None
        self.biases = None

        self.train_predictions = None
        self.test_predictions = None
        self.validate_predictions = None

        self.train_accuracy = None
        self.validate_accuracy = None
        self.test_accuracy = None

        # Best validation accuracy seen so far.
        self.best_validation_accuracy = 0.0

        # Iteration-number for last improvement to validation accuracy.
        self.last_improvement = 0

        # In order to save the variables of the neural network, we now create a so-called Saver-object which is used
        # for storing and retrieving all the variables of the TensorFlow graph. Nothing is actually saved at this
        # point, which will be done further below in the optimize()-function.
        self.saver = None

        # Directory where all trained models will be stored
        self.save_dir = os.path.join(os.path.dirname(__file__), '..', 'trained_models')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def configure(self, configuration_context):
        """
        Principle entry point into all BaseNeuralNetworks.

        :param configuration_context: An instance of ConfigurationContext
        :return:
        """
        self.config = configuration_context

    def generate(self):
        """
        Principle entry point into the network.

        Invokes the following methods in this order:

          1. #fit           - Does the heavy lifting of training the network
          2. #validate      - Checks the accuracy of the model against the validation set
          3. #predict       - Checks the accuracy of the model against the test set
          4. #serialize     - Serialize the entire dataset.
          5. #persist

        :return: None
        """
        os.system('say "Model fit started"')
        [self.__with_time(op['label'], op['callback']) for op in [
            # {'label': 'START TENSORFLOW SESSION', 'callback': self.__open_session},
            {'label': 'FIT MODEL', 'callback': self.fit},
            {'label': 'SERIALIZE TRAINED MODEL', 'callback': self.serialize},
            {'label': 'PERSIST SERIALIZED TRAINED MODEL', 'callback': self.__persist},
            # {'label': 'START TENSORFLOW SESSION', 'callback': self.__close_session}
        ]]
        os.system('say "Model fit complete!"')
        os.system('say "Network serialized to the data directory."')
        os.system('say "Best validation accuracy achieved was {:.002f} percent at iteration {}."'.format(
            (self.best_validation_accuracy * 100), self.last_improvement))
        os.system('say "The most accurate validation model has been serialized to the trained models directory."')

    def evaluate_accuracy(self, session, validation_accuracy, total_iterations):
        if validation_accuracy > 0.85 and validation_accuracy > self.best_validation_accuracy:
            if self.saver is None:
                self.saver = tf.train.Saver()

            # Update the best-known validation accuracy.
            self.best_validation_accuracy = validation_accuracy

            # Set the iteration for the last improvement to current.
            self.last_improvement = total_iterations

            # Save all variables of the TensorFlow graph to file.
            save_path = os.path.join(self.save_dir, self.__generate_file_name())

            self.saver.save(sess=session, save_path=save_path)
            return True
        return False

    def __with_time(self, label, callback):
        start = datetime.now()
        print('')
        print('')
        print("===========> [{}] Started at {}".format(label, start.time()))
        print('')
        print('')

        callback()

        end = datetime.now()
        print('')
        print('')
        print("===========> [{}] Finished at {}".format(label, end.time()))
        print('')
        print("===========> [{}] Wall time: {}".format(label, end - start))
        print('')
        print("└[∵┌]   └[ ∵ ]┘   [┐∵]┘   └[ ∵ ]┘   └[∵┌]   └[ ∵ ]┘   [┐∵]┘   └[ ∵ ]┘   └[∵┌]")
        print('')
        print('')
        print('')

    def __generate_file_name(self):
        return '{}_{}_best_validation_{}S_{}LR_{}E_{}B'.format(
            self.__class__.__name__,
            self.uuid,
            "{:.002f}".format(self.config.data.split_size),
            "{:.004f}".format(self.config.hyper_parameters.start_learning_rate),
            self.config.hyper_parameters.epochs,
            self.config.hyper_parameters.batch_size)

    def predict(self):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError

    def serialize(self, data={}):
        return {
            **data,
            **{
                'cost': self.cost,
                'weights': self.weights,
                'biases': self.biases,
                'config': {
                    'hyper_parameters': self.config.hyper_parameters.__dict__,
                    'data': self.config.data.serialize()
                },
                'predictions': {
                    'train': self.train_predictions,
                    'validate': self.validate_predictions,
                    'test': self.test_predictions
                },
                'accuracy': {
                    'train': self.train_accuracy,
                    'validate': self.validate_accuracy,
                    'test': self.test_accuracy
                }
            }
        }

    def __persist(self):
        if self.best_validation_accuracy > 0.85:
            TrainedDataSerializer.save_data(
                data=self.serialize(),
                pickle_file='{}_trained_{}TA_{}VA_{}TestA_{}S_{}sLR_{}eLR_{}E_{}B.pickle'.format(
                    self.__class__.__name__,
                    "{:.004f}".format(self.train_accuracy),
                    "{:.004f}".format(self.validate_accuracy),
                    "{:.004f}".format(self.test_accuracy),
                    "{:.004f}".format(self.config.data.split_size),
                    "{:.004f}".format(self.config.hyper_parameters.start_learning_rate),
                    "{:.004f}".format(self.config.hyper_parameters.end_learning_rate),
                    self.config.hyper_parameters.epochs,
                    self.config.hyper_parameters.batch_size),
                overwrite=True
            )

    # Helper functions

    def predict_cls(self, images, labels, cls_true, session, logits, features_placeholder, labels_placeholder):
        # Number of images.
        num_images = len(images)

        # Allocate an array for the predicted classes which
        # will be calculated in batches and filled into this array.
        cls_pred = np.zeros(shape=num_images, dtype=np.int)

        # Now calculate the predicted classes for the batches.
        # We will just iterate through all the batches.
        # There might be a more clever and Pythonic way of doing this.

        # The starting index for the next batch is denoted i.
        i = 0

        while i < num_images:
            # The ending index for the next batch is denoted j.
            j = min(i + self.config.hyper_parameters.batch_size, num_images)

            # Create a feed-dict with the images and labels
            # between index i and j.
            feed_dict = {features_placeholder: images[i:j, :], labels_placeholder: labels[i:j, :]}

            # Calculate the predicted class using TensorFlow.
            y_pred_cls = tf.argmax(logits, dimension=1)
            y_true_cls = tf.argmax(labels_placeholder, 1)
            correct_prediction = tf.equal(y_pred_cls, y_true_cls)
            cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

            # Set the start-index for the next batch to the
            # end-index of the current batch.
            i = j

        # Create a boolean array whether each image is correctly classified.
        correct = (cls_true == cls_pred)

        return correct, cls_pred

    # def __open_session(self):
    #     self.session = tf.InteractiveSession()

    # def __close_session(self):
    #     self.session.close()

    def __str__(self):
        result = []
        result.append(' ')
        for k, v in self.__dict__.items():
            result.append('{} - {}'.format(k, str(v)))
        result.append(' ')
        return '\n'.join(result)
