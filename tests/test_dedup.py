import numpy as np

from organizeme import dedup


def test_union_find_groups_connected_components():
    uf = dedup._UnionFind(['a', 'b', 'c', 'd'])
    uf.union('a', 'b')
    uf.union('b', 'c')
    groups = sorted([sorted(g) for g in uf.groups()])
    assert ['a', 'b', 'c'] in groups
    assert ['d'] in groups


def test_near_edges_links_similar_vectors():
    paths = ['p0', 'p1', 'p2']
    vecs = np.asarray(
        [[1.0, 0.0], [0.999, 0.0447], [0.0, 1.0]], dtype='float32'
    )
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    edges = dedup._near_edges(paths, vecs, threshold=0.95)
    linked = {frozenset({a, b}) for a, b, _ in edges}
    assert frozenset({'p0', 'p1'}) in linked
    assert frozenset({'p0', 'p2'}) not in linked


def test_exact_duplicates_detected(tmp_path, monkeypatch):
    # No embeddings needed: identical bytes must cluster via the hash stage.
    monkeypatch.setattr(dedup.embeddings, 'embed_texts', lambda texts: (np.zeros((0, 0), 'float32'), []))
    monkeypatch.setattr(dedup.embeddings, 'embed_images', lambda paths: (np.zeros((0, 0), 'float32'), []))

    (tmp_path / 'a.txt').write_text('identical content')
    (tmp_path / 'b.txt').write_text('identical content')
    (tmp_path / 'c.txt').write_text('something else')

    clusters = dedup.find_clusters(str(tmp_path))
    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster['method'] == 'exact'
    assert cluster['similarity'] == 1.0
    assert len(cluster['duplicates']) == 1


def test_detect_duplicates_quarantines_on_apply(tmp_path, monkeypatch):
    monkeypatch.setattr(dedup.embeddings, 'embed_texts', lambda texts: (np.zeros((0, 0), 'float32'), []))
    monkeypatch.setattr(dedup.embeddings, 'embed_images', lambda paths: (np.zeros((0, 0), 'float32'), []))

    (tmp_path / 'a.txt').write_text('dup')
    (tmp_path / 'b.txt').write_text('dup')

    report = dedup.detect_duplicates(str(tmp_path), apply=True)
    assert report['duplicate_count'] == 1
    assert report['applied'] is True
    quarantined = list((tmp_path / '_duplicates').iterdir())
    assert len(quarantined) == 1
