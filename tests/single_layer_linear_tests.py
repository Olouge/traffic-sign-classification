from classifiers.base_neural_network import ConfigurationContext
from classifiers.linear.single_layer_linear import SingleLayerLinear, SingleLayerHyperParametersContext
from datasets.german_traffic_signs import GermanTrafficSignDataset

# Create fresh German Traffic Sign dataset
data = GermanTrafficSignDataset()
data.configure(one_hot=True, train_validate_split_percentage=0.20)


# [TEST] Simple Neural Network
def test_training(hidden_layer_neuron_count=512, start_learning_rate=0.2, epochs=1, batch_size=20,
                  required_accuracy_improvement=50):
    # hyper_parameters = HyperParametersContext(start_learning_rate=0.2, epochs=1000, batch_size=20, required_accuracy_improvement=50)
    hyper_parameters = SingleLayerHyperParametersContext(hidden_layer_neuron_count=hidden_layer_neuron_count,
                                                         start_learning_rate=start_learning_rate,
                                                         epochs=epochs,
                                                         batch_size=batch_size,
                                                         required_accuracy_improvement=required_accuracy_improvement)
    config = ConfigurationContext(data, hyper_parameters)

    simple_nn = SingleLayerLinear()
    simple_nn.configure(config)
    simple_nn.generate()

    # print(simple_nn)

    return simple_nn


# simple_nn = test_training(hidden_layer_neuron_count=512, start_learning_rate=0.22, epochs=200, batch_size=32, required_accuracy_improvement=50)


def test_predictions(checkpoint):
    data = GermanTrafficSignDataset()
    data.configure(one_hot=True, train_validate_split_percentage=0.001)

    print('')
    print('Prediction Accuracy for {}'.format(checkpoint))
    simple_nn = SingleLayerLinear()
    simple_nn.configure(ConfigurationContext(data, SingleLayerHyperParametersContext()))
    simple_nn.predict(images=data.predict_flat, true_labels=data.predict_labels, model_name=checkpoint)


# test_predictions(simple_nn.save_path())
# test_predictions('SingleLayerLinear_34f0f1f9-1b64-4eef-a6b2-d18d2830e7de_best_validation_0.05S_0.2200LR_200E_32B')
# test_predictions('SingleLayerLinear_c4faddf8-4457-45ac-9c1f-1b5444be235a_best_validation_0.918395VA_0.15S_0.2200LR_300E_32B')
test_predictions('SingleLayerLinear_38eb4c21-45f6-4695-a257-6f964ffef68f_best_validation_0.20S_0.2200LR_200E_32B')

# test_predictions('SingleLayerLinear_02cd0732-d0f0-497e-977d-3faa35033d67_best_validation_0.20S_0.2000LR_500E_20B')

# test_predictions('SingleLayerLinear_6a528a5e-cf0d-4ade-a8ec-6c8e6aeb8971_best_validation_0.20S_0.2000LR_500E_20B')
# test_predictions('SingleLayerLinear_f5604170-7cef-480c-a4a8-6fe8ce78f7c5_best_validation_0.20S_0.2000LR_100E_20B')
# test_predictions('SingleLayerLinear_59eda874-2632-409b-814a-5c373c06f43b_best_validation_0.20S_0.2000LR_1000E_20B')
# test_predictions('SingleLayerLinear_best_validation_0.2000S_0.2000LR_1000E_20B')
