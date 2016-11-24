from datasets.german_traffic_sign_dataset import GermanTrafficSignDataset

def test_configure(data):
    data.configure()

def test_resume(data):
    data.resume()

def test_print(data):
    print(data)

def test_persist(data):
    data.persist(overwrite=True)

def test_restore(data):
    data.restore()


print('[TEST] Configure from source file')
print('')

data = GermanTrafficSignDataset()

test_configure(data)
test_print(data)
test_persist(data)

del data

# test_restore(data)

print('[TEST] Resume from persisted file')
print('')
print('')
print('')


data = GermanTrafficSignDataset()

test_resume(data)
test_print(data)