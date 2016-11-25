import tensorflow as tf
import numpy as np


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


class BaseNeuralNetwork:
    def __init__(self, german_traffic_sign_dataset, hyper_parameters=None):
        """
        :param german_traffic_sign_dataset: An instance of datasets.GermanTrafficSignDataset
        :param hyper_parameters: An instance of HyperParametersContext
        """
        self.data = german_traffic_sign_dataset
        if hyper_parameters is None:
            self.hyper_parameters = HyperParametersContext()
        else:
            self.hyper_parameters = hyper_parameters

        self.cross_entropy = None
        self.weights = None
        self.biases = None
        self.accuracy = None
        self.correct_prediction = None

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
        [f() for f in [
            self.fit,
            self.__serialize,
            self.__persist
        ]]

    def predict(self):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError

    def __serialize(self, data={}):
        return self.data.__serialize(
            {
                **data,
                **{
                    'learning_rate': self.hyper_parameters.learning_rate,
                    'epochs': self.hyper_parameters.epochs,
                    'batch_size': self.hyper_parameters.batch_size
                }
            }
        )

    def __persist(self):
        self.data.persist(
            data=self.__serialize(),
            pickle_name='SimpleNeuralNetworkClassifier_{}_{}_{}_{}.pickle'.format(self.data.split_size,
                                                                                  self.hyper_parameters.learning_rate,
                                                                                  self.hyper_parameters.epochs,
                                                                                  self.hyper_parameters.batch_size),
            overwrite=True
        )

    def __str__(self):
        result = []
        result.append(' ')
        for k, v in self.__dict__.items():
            result.append('{} - {}'.format(k, str(v)))
        result.append(' ')
        return '\n'.join(result)
