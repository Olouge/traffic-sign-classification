from pipelines.german_traffic_signs import GermanTrafficSignDataset

data = GermanTrafficSignDataset()
print(data)


def test_persist(data):
    data.persist(overwrite=True)

def test_restore(data):
    data.restore()


test_persist(data)
test_restore(data)