import numpy as np

from classifiers.base_neural_network import ConfigurationContext
from classifiers.linear.single_layer_linear import SingleLayerLinear, SingleLayerHyperParametersContext
from zimpy.datasets.cifar10 import Cifar10Dataset
from plot.image_plotter import ImagePlotter


def generate_dataset(one_hot=True,
                     train_validate_split_percentage=0.2):
    data = Cifar10Dataset()
    data.configure(one_hot=one_hot, train_validate_split_percentage=train_validate_split_percentage)
    return data


def compute_top_k(trained_model_name, hidden_layer_neuron_count, k=5):
    data = generate_dataset()

    print('')
    print('Top 5 for {}'.format(trained_model_name))

    simple_nn = SingleLayerLinear()

    hyper_parameters = SingleLayerHyperParametersContext(hidden_layer_neuron_count=hidden_layer_neuron_count)
    config_context = ConfigurationContext(dataset=data, hyper_parameters=hyper_parameters)

    simple_nn.configure(config_context)

    top_k = simple_nn.top_k(k=k, x=data.predict_flat, y=data.predict_labels, model_name=trained_model_name)

    # print(top_k)

    return top_k

def visualize_top_k(top_k):
    print('')
    print('Visualize top_k:')
    # print(top_k)


def test_predictions(checkpoint, hidden_layer_neuron_count=512):
    data = generate_dataset(one_hot=True, train_validate_split_percentage=0.2)

    print('')
    print('Prediction Accuracy for {}'.format(checkpoint))
    simple_nn = SingleLayerLinear()
    simple_nn.configure(ConfigurationContext(dataset=data, hyper_parameters=SingleLayerHyperParametersContext(
        hidden_layer_neuron_count=hidden_layer_neuron_count)))
    correct, predicted = simple_nn.predict(images=data.predict_flat, true_labels=data.predict_labels,
                                           model_name=checkpoint)
    print(correct)
    print(predicted)

    ImagePlotter.plot_images(data.predict_orig[:12], np.argmax(data.predict_labels[:12], axis=1), cls_pred=predicted,
                             rows=2, columns=6)


# Create fresh German Traffic Sign dataset
# data = generate_dataset(one_hot=True, train_validate_split_percentage=0.2)

# Top 5

# compute_top_k(trained_model_name='SingleLayerLinear_38eb4c21-45f6-4695-a257-6f964ffef68f_best_validation_0.20S_0.2200LR_200E_32B', hidden_layer_neuron_count=512)

top_k = compute_top_k(k=5,
              trained_model_name='SingleLayerLinear_e536a998-bc7c-461c-8b2a-fc278de569d8_best_validation_0.20S_0.2000LR_300E_20B_256HN',
              hidden_layer_neuron_count=256)

visualize_top_k(top_k)



# accuracies
# test_predictions('SingleLayerLinear_e536a998-bc7c-461c-8b2a-fc278de569d8_best_validation_0.20S_0.2000LR_300E_20B_256HN', hidden_layer_neuron_count=256)
