from classifiers.base_neural_network import HyperParametersContext, ConfigurationContext
from classifiers.simple_neural_network import SimpleNeuralNetwork
from datasets.german_traffic_signs import GermanTrafficSignDataset

# Create fresh German Traffic Sign dataset
data = GermanTrafficSignDataset()
data.configure(one_hot=True, train_validate_split_percentage=0.05)

# [TEST] Simple Neural Network
hyper_parameters = HyperParametersContext(learning_rate=0.25, epochs=200, batch_size=32)
config = ConfigurationContext(data, hyper_parameters)

simple_nn = SimpleNeuralNetwork()
simple_nn.configure(config)
simple_nn.generate()

# print(simple_nn)
