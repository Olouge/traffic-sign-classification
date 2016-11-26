import tensorflow as tf

from classifiers.base_neural_network import BaseNeuralNetwork


class TemplateNeuralNetwork(BaseNeuralNetwork):
    def fit(self):
        raise NotImplementedError
