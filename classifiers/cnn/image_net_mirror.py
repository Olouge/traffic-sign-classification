import tensorflow as tf

from classifiers.base_neural_network import BaseNeuralNetwork, HyperParametersContext


class ImageNetMirrorHyperParameterContext(HyperParametersContext):
    def __init__(
            self,
            **kwargs
    ):
        super(ImageNetMirrorHyperParameterContext, self).__init__(**kwargs)

class ImageNetMirror(BaseNeuralNetwork):
    def fit(self):
        print(self)
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError