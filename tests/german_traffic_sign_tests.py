from datasets.german_traffic_sign_dataset import GermanTrafficSignDataset

def test_configure(data):
    data.configure()

def test_resume(data):
    data.restore()

def test_print(data):
    print(data)
    print('train_lables:')
    print(data.train_labels[0:1])

def test_persist(data):
    data.persist(overwrite=True)

def test_restore(data):
    data.restore()


print('[TEST] Configure from source file (non-encoded labels)')
print('')

data = GermanTrafficSignDataset(one_hot=False)

test_configure(data)
test_print(data)
test_persist(data)

del data

print('')
print('')
print('')
print('[TEST] Configure from source file with one-hot encoded labels')
print('')

data = GermanTrafficSignDataset()

test_configure(data)
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

test_resume(data)
test_print(data)