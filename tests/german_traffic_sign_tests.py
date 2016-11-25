from datasets.german_traffic_signs import GermanTrafficSignDataset

def test_configure(data, one_hot=True, train_validate_split_percentage=0.05):
    data.configure(one_hot=one_hot, train_validate_split_percentage=train_validate_split_percentage)

def test_print(data):
    print(data)
    print('train_lables:')
    print(data.train_labels[0:1])

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