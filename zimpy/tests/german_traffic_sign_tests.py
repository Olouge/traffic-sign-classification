import numpy as np

from zimpy.datasets.german_traffic_signs import GermanTrafficSignDataset

def test_configure(data, one_hot=True, train_validate_split_percentage=0.05):
    data.configure(one_hot=one_hot, train_validate_split_percentage=train_validate_split_percentage)

def test_dimensions(data):
    assert len(data.train_orig) == len(data.train_labels), 'train features and labels must be the same size'
    assert len(data.validate_orig) == len(data.validate_labels), 'validate features and labels must be the same size'
    assert len(data.test_orig) == len(data.test_labels), 'test features and labels must be the same size'
    assert len(data.predict_orig) == len(data.predict_labels), 'predict features and labels must be the same size'

def test_print(data):
    print(data)

    for bin_name, bin_data in {
        'train': {'features': data.train_orig, 'labels': data.train_labels},
        'validate': {'features': data.validate_orig, 'labels': data.validate_labels},
        'test': {'features': data.test_orig, 'labels': data.test_labels},
        'predict': {'features': data.predict_orig, 'labels': data.predict_labels}}.items():

        perm = np.arange(len(bin_data['labels']))
        np.random.shuffle(perm)
        idx = perm[0]
        label, sign_name = data.label_sign_name(bin_data['labels'], idx)
        print(bin_name, 'label', idx, ':', label, '-', data.sign_names_map[label])

def test_persist(data):
    data.persist(data.serialize(), overwrite=True)

def test_restore(data):
    data.restore()


print('[TEST] Configure from source file (non-encoded labels)')
print('')

data = GermanTrafficSignDataset()

test_configure(data, one_hot=False)
test_dimensions(data)
test_print(data)
test_persist(data)

del data

print('')
print('')
print('')
print('[TEST] Configure from source file with one-hot encoded labels')
print('')

data = GermanTrafficSignDataset()

test_configure(data, one_hot=True)
test_dimensions(data)
test_print(data)
test_persist(data)

del data



print('')
print('')
print('')
print('[TEST] Resume from persisted file')
print('')
print('')
print('')


data = GermanTrafficSignDataset()
test_dimensions(data)
test_restore(data)
test_print(data)