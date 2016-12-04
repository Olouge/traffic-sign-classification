import numpy as np
import matplotlib.pyplot as plt

from classifiers.base_neural_network import ConfigurationContext
from classifiers.linear.single_layer_linear import SingleLayerLinear, SingleLayerHyperParametersContext
from datasets.german_traffic_signs import GermanTrafficSignDataset
from plot.image_plotter import ImagePlotter


def generate_dataset(one_hot=True,
                     train_validate_split_percentage=0.2):
    data = GermanTrafficSignDataset()
    data.configure(one_hot=one_hot, train_validate_split_percentage=train_validate_split_percentage)
    return data


def test_training(one_hot=True, train_validate_split_percentage=0.2,
                  optimizer_type=ConfigurationContext.OPTIMIZER_TYPE_GRADIENT_DESCENT,
                  hidden_layer_neuron_count=512,
                  start_learning_rate=0.22,
                  training_epochs=200,
                  batch_size=32,
                  required_accuracy_improvement=50):
    data = generate_dataset(one_hot=one_hot, train_validate_split_percentage=train_validate_split_percentage)

    hyper_parameters = SingleLayerHyperParametersContext(hidden_layer_neuron_count=hidden_layer_neuron_count,
                                                         start_learning_rate=start_learning_rate,
                                                         training_epochs=training_epochs,
                                                         batch_size=batch_size,
                                                         required_accuracy_improvement=required_accuracy_improvement)
    config = ConfigurationContext(dataset=data, optimizer_type=optimizer_type, hyper_parameters=hyper_parameters)

    simple_nn = SingleLayerLinear()
    simple_nn.configure(config)
    simple_nn.generate()

    # Plot the losses over time
    # plt.plot(simple_nn.losses)
    # plt.show()

    # print(simple_nn)

    return simple_nn


def test_accuracies(checkpoint, hidden_layer_neuron_count=512):
    data = GermanTrafficSignDataset()
    data.configure(one_hot=True, train_validate_split_percentage=0.2)

    print('')
    print('Accuracies for {}'.format(checkpoint))

    simple_nn = SingleLayerLinear()
    simple_nn.configure(ConfigurationContext(dataset=data, hyper_parameters=SingleLayerHyperParametersContext(
        hidden_layer_neuron_count=hidden_layer_neuron_count)))

    # simple_nn.predict(images=data.train_flat, true_labels=data.train_labels, model_name=checkpoint)
    # simple_nn.predict(images=data.validate_flat, true_labels=data.validate_labels, model_name=checkpoint)
    # simple_nn.predict(images=data.test_flat, true_labels=data.test_labels, model_name=checkpoint)
    simple_nn.predict(images=data.predict_flat, true_labels=data.predict_labels, model_name=checkpoint)


def top_5(checkpoint, hidden_layer_neuron_count):
    data = GermanTrafficSignDataset()
    data.configure(one_hot=True, train_validate_split_percentage=0.2)

    print('')
    print('Accuracies for {}'.format(checkpoint))

    simple_nn = SingleLayerLinear()
    simple_nn.configure(ConfigurationContext(dataset=data, hyper_parameters=SingleLayerHyperParametersContext(
        hidden_layer_neuron_count=hidden_layer_neuron_count)))

    simple_nn.top_k(images=data.predict_flat, true_labels=data.predict_labels, model_name=checkpoint)


def test_predictions(checkpoint, hidden_layer_neuron_count=512):
    data = GermanTrafficSignDataset()
    data.configure(one_hot=True, train_validate_split_percentage=0.001)

    print('')
    print('Prediction Accuracy for {}'.format(checkpoint))
    simple_nn = SingleLayerLinear()
    simple_nn.configure(ConfigurationContext(dataset=data, hyper_parameters=SingleLayerHyperParametersContext(
        hidden_layer_neuron_count=hidden_layer_neuron_count)))
    correct, predicted = simple_nn.predict(images=data.predict_flat, true_labels=data.predict_labels, model_name=checkpoint)
    print(correct)
    print(predicted)

    ImagePlotter.plot_images(data.predict_orig[:12], np.argmax(data.predict_labels[:12], axis=1), cls_pred=predicted, rows=2, columns=6)


# Create fresh German Traffic Sign dataset
# data = generate_dataset(one_hot=True, train_validate_split_percentage=0.2)

# Top 5

# top_5('SingleLayerLinear_38eb4c21-45f6-4695-a257-6f964ffef68f_best_validation_0.20S_0.2200LR_200E_32B', hidden_layer_neuron_count=512)
# top_5('SingleLayerLinear_e536a998-bc7c-461c-8b2a-fc278de569d8_best_validation_0.20S_0.2000LR_300E_20B_256HN', hidden_layer_neuron_count=256)

# GradientDescent optimizer

# Adagrad optimizer
# Yielded the following trained network:
#
# Epoch: 0197 of 0200
#   loss:               3.842783213
#   batch accuracy:     1.0
#   train accuracy:     0.971467
#   validate accuracy:  0.893139
#   test accuracy:      0.748852
#   predict accuracy:   0.692308
#   batch size:         20
#   learning rate:      0.2
#
# No improvement found in a while, stopping optimization after 197 iterations. Final accuracy, 89% at iteration 171.
#
# simple_nn_gd = test_training(
#     one_hot=True,
#     train_validate_split_percentage=0.2,
#     optimizer_type=ConfigurationContext.OPTIMIZER_TYPE_ADAGRAD,
#     training_epochs=200,
#     batch_size=20,
#     required_accuracy_improvement=25,
#     start_learning_rate=2e-1,
#     hidden_layer_neuron_count=512
# )

# simple_nn_gd = test_training(dataset=data, optimizer_type=ConfigurationContext.OPTIMIZER_TYPE_GRADIENT_DESCENT, hidden_layer_neuron_count=512, start_learning_rate=0.2, training_epochs=200, batch_size=32)
# simple_nn_gd = test_training(dataset=data, optimizer_type=ConfigurationContext.OPTIMIZER_TYPE_GRADIENT_DESCENT, hidden_layer_neuron_count=256, start_learning_rate=0.2, training_epochs=2, batch_size=20)

# Adagrad optimizer
# simple_nn_ag = test_training(dataset=data, optimizer_type=ConfigurationContext.OPTIMIZER_TYPE_ADAGRAD, hidden_layer_neuron_count=512, start_learning_rate=0.22, training_epochs=200, batch_size=20)



# accuracies
# test_accuracies('SingleLayerLinear_dca487dd-bcec-4547-a051-8495a943bcf7_best_validation_0.20S_0.2000LR_300E_20B_512HN', hidden_layer_neuron_count=512)
# test_predictions('SingleLayerLinear_e536a998-bc7c-461c-8b2a-fc278de569d8_best_validation_0.20S_0.2000LR_300E_20B_256HN', hidden_layer_neuron_count=256)
# test_accuracies('SingleLayerLinear_99d2e377-a8d6-4c22-89b3-13eb9edea291_best_validation_0.20S_0.2000LR_500E_32B', hidden_layer_neuron_count=512)

# 100% accuracy for 5 german signs; 70% with my own images thrown in there
# test_accuracies('SingleLayerLinear_38eb4c21-45f6-4695-a257-6f964ffef68f_best_validation_0.20S_0.2200LR_200E_32B', hidden_layer_neuron_count=512)

# 70% with my own images thrown in there
# test_accuracies('SingleLayerLinear_fa4c9d40-ce78-4d60-be1f-e852617caa12_best_validation_0.20S_0.2200LR_200E_20B', hidden_layer_neuron_count=256)

# test_accuracies('SingleLayerLinear_faeefc98-c391-4243-901f-716718fd63d7_best_validation_0.20S_0.2200LR_200E_20B', hidden_layer_neuron_count=256)


# predictions
# test_predictions(simple_nn.save_path())
# test_predictions('SingleLayerLinear_34f0f1f9-1b64-4eef-a6b2-d18d2830e7de_best_validation_0.05S_0.2200LR_200E_32B')
# test_predictions('SingleLayerLinear_c4faddf8-4457-45ac-9c1f-1b5444be235a_best_validation_0.918395VA_0.15S_0.2200LR_300E_32B')
# test_predictions('SingleLayerLinear_38eb4c21-45f6-4695-a257-6f964ffef68f_best_validation_0.20S_0.2200LR_200E_32B', hidden_layer_neuron_count=512)
# test_predictions('SingleLayerLinear_faeefc98-c391-4243-901f-716718fd63d7_best_validation_0.20S_0.2200LR_200E_20B', hidden_layer_neuron_count=256)

# test_predictions('SingleLayerLinear_02cd0732-d0f0-497e-977d-3faa35033d67_best_validation_0.20S_0.2000LR_500E_20B')

# test_predictions('SingleLayerLinear_6a528a5e-cf0d-4ade-a8ec-6c8e6aeb8971_best_validation_0.20S_0.2000LR_500E_20B')
# test_predictions('SingleLayerLinear_f5604170-7cef-480c-a4a8-6fe8ce78f7c5_best_validation_0.20S_0.2000LR_100E_20B')
# test_predictions('SingleLayerLinear_59eda874-2632-409b-814a-5c373c06f43b_best_validation_0.20S_0.2000LR_1000E_20B')
# test_predictions('SingleLayerLinear_best_validation_0.2000S_0.2000LR_1000E_20B')
