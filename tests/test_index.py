import numpy as np

from organizeme.index import VectorIndex


def _normed(rows):
    arr = np.asarray(rows, dtype='float32')
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def test_search_returns_nearest_first():
    idx = VectorIndex(3)
    vecs = _normed([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    idx.add(vecs, [{'path': 'x'}, {'path': 'y'}, {'path': 'z'}])
    results = idx.search(vecs[0], k=2)
    assert results[0][1]['path'] == 'x'
    assert results[0][0] > results[1][0]


def test_save_and_load_round_trip(tmp_path):
    idx = VectorIndex(3)
    vecs = _normed([[1, 0, 0], [0, 1, 0]])
    idx.add(vecs, [{'path': 'a'}, {'path': 'b'}])
    idx.save(str(tmp_path / 'idx'))

    loaded = VectorIndex.load(str(tmp_path / 'idx'))
    assert len(loaded) == 2
    assert loaded.metadata[0]['path'] == 'a'
    assert loaded.search(vecs[1], k=1)[0][1]['path'] == 'b'


def test_empty_index_search_returns_empty():
    assert VectorIndex(4).search(np.ones(4, dtype='float32')) == []
