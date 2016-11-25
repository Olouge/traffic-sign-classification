import datetime

import tensorflow as tf
import numpy as np

from serializers.trained_data_serializer import TrainedDataSerializer


class HyperParametersContext:
    def __init__(
            self,
            learning_rate=0.2,
            epochs=15,
            batch_size=20,
            exponential_decay_config=None
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.exponential_decay_config = exponential_decay_config

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
        self.config = None

        self.cross_entropy = None
        self.weights = None
        self.biases = None

        self.train_predictions = None
        self.test_predictions = None
        self.validate_predictions = None

        self.train_accuracy = None
        self.validate_accuracy = None
        self.test_accuracy = None

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
        [self.__with_time(op['label'], op['callback']) for op in [
            {'label': 'FIT MODEL', 'callback': self.fit},
            {'label': 'SERIALIZE TRAINED MODEL', 'callback': self.serialize},
            {'label': 'PERSIST SERIALIZED TRAINED MODEL', 'callback': self.__persist}
        ]]

    def __with_time(self, label, callback):
        start = datetime.datetime.now()
        print('')
        print('')
        print("===========> [{}] Started at {}".format(label, start.time()))
        print('')
        print('')

        callback()

        end = datetime.datetime.now()
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

    def predict(self):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError

    def serialize(self, data={}):
        return {
            **data,
            **{
                'cross_entropy': self.cross_entropy,
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
        TrainedDataSerializer.save_data(
            data=self.serialize(),
            pickle_file='SimpleNeuralNetworkClassifier_{}S_{}LR_{}E_{}B.pickle'.format(
                int(self.config.data.split_size * 100),
                int(
                    self.config.hyper_parameters.learning_rate * 100),
                self.config.hyper_parameters.epochs,
                self.config.hyper_parameters.batch_size),
            overwrite=True
        )

    def __str__(self):
        result = []
        result.append(' ')
        for k, v in self.__dict__.items():
            result.append('{} - {}'.format(k, str(v)))
        result.append(' ')
        return '\n'.join(result)
