from classifiers.base_neural_network import ConfigurationContext
from classifiers.cnn.image_net_mirror import ImageNetMirrorHyperParameterContext, ImageNetMirror
from datasets.german_traffic_signs import GermanTrafficSignDataset


def generate_dataset(one_hot=True,
                     train_validate_split_percentage=0.2):
    data = GermanTrafficSignDataset()
    data.configure(one_hot=one_hot, train_validate_split_percentage=train_validate_split_percentage)
    return data


def test_training(one_hot=True, train_validate_split_percentage=0.2,
                  optimizer_type=ConfigurationContext.OPTIMIZER_TYPE_GRADIENT_DESCENT,
                  start_learning_rate=0.22,
                  epochs=200,
                  batch_size=32,
                  required_accuracy_improvement=50):
    data = generate_dataset(one_hot=one_hot, train_validate_split_percentage=train_validate_split_percentage)

    hyper_parameters = ImageNetMirrorHyperParameterContext(
        start_learning_rate=start_learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        required_accuracy_improvement=required_accuracy_improvement
    )
    config = ConfigurationContext(dataset=data, optimizer_type=optimizer_type, hyper_parameters=hyper_parameters)

    cnn_clf = ImageNetMirror()
    cnn_clf.configure(config)
    cnn_clf.generate()

    # Plot the losses over time
    # plt.plot(simple_nn.losses)
    # plt.show()

    # print(simple_nn)

    return cnn_clf


print(test_training())
