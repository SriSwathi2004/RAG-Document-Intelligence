import numpy as np
from src.embeddings import HashingEmbedder


def test_shape_and_unit_norm():
    v = HashingEmbedder(128).embed(["hello world", "machine learning"])
    assert v.shape == (2, 128)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


def test_similar_text_scores_higher():
    e = HashingEmbedder()
    a, b, c = e.embed(["random forest trees vote", "a forest of decision trees votes", "http status code 429"])
    assert a @ b > a @ c


def test_empty_list():
    assert HashingEmbedder().embed([]).shape[0] == 0
