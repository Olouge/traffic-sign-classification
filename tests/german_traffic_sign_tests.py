import numpy as np

from datasets.german_traffic_signs import GermanTrafficSignDataset

def test_configure(data, one_hot=True, train_validate_split_percentage=0.05):
    data.configure(one_hot=one_hot, train_validate_split_percentage=train_validate_split_percentage)

def test_print(data):
    print(data)
    idx = 0
    for bin_name, bin_data in {'train': {'features': data.train_orig, 'labels': data.train_labels}, 'validate': {'features': data.validate_orig, 'labels': data.validate_labels}, 'test': {'features': data.test_orig, 'labels': data.test_labels}}.items():
        # image = bin_data['features'][idx]
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

test_restore(data)
test_print(data)