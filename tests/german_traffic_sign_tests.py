from german_traffic_signs import GermanTrafficSignDataset
from german_traffic_signs import TrainedDataMarshaler

data = GermanTrafficSignDataset()
print(data)


def test_persist(data):
    data.persist()

def test_restore(data):
    data.restore()


test_persist(data)
test_restore(data)