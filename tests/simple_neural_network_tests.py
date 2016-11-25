from classifiers.base_neural_network import HyperParametersContext
from classifiers.simple_neural_network import SimpleNeuralNetwork
from datasets.german_traffic_sign_dataset import GermanTrafficSignDataset

# [TEST] Vanilla hyperparameters
data = GermanTrafficSignDataset()
data.configure()

hyper_parameters = HyperParametersContext(learning_rate=0.2,epochs=50, batch_size=20)
simple_nn = SimpleNeuralNetwork(data, hyper_parameters)
simple_nn.generate()
print(simple_nn)

