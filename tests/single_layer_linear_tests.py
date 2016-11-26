from classifiers.base_neural_network import HyperParametersContext, ConfigurationContext
from classifiers.linear.single_layer_linear import SingleLayerLinear
from datasets.german_traffic_signs import GermanTrafficSignDataset

# Create fresh German Traffic Sign dataset
data = GermanTrafficSignDataset()
data.configure(one_hot=True, train_validate_split_percentage=0.20)

# [TEST] Simple Neural Network
# hyper_parameters = HyperParametersContext(start_learning_rate=0.2, epochs=1000, batch_size=20, required_accuracy_improvement=50)
hyper_parameters = HyperParametersContext(start_learning_rate=0.2, epochs=1, batch_size=20, required_accuracy_improvement=50)
config = ConfigurationContext(data, hyper_parameters)

simple_nn = SingleLayerLinear()
simple_nn.configure(config)
simple_nn.generate()

# print(simple_nn)
