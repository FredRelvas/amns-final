import numpy as np

from pcacluster.data import Preprocessor, load_dataset


def test_preprocessor_uses_train_statistics_only():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(50, 4))
    X_test = rng.normal(loc=100, size=(20, 4))
    pre = Preprocessor().fit(X_train)

    Z_train = pre.transform(X_train)
    assert np.allclose(Z_train.mean(axis=0), 0)
    assert np.allclose(Z_train.std(axis=0), 1)
    # test is shifted by the train mean, not re-centered on itself
    assert pre.transform(X_test).mean() > 10


def test_preprocessor_drops_columns_constant_in_train():
    X_train = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    X_test = np.array([[1.0, 7.0]])
    assert Preprocessor().fit(X_train).transform(X_test).shape == (1, 1)


def test_load_digits_subsample():
    X, y = load_dataset("digits", n=100)
    assert X.shape == (100, 64) and y.shape == (100,)


def test_synthetic_shape_and_clusters():
    X, y = load_dataset("synthetic", n=300, p=20, n_clusters=4)
    assert X.shape == (300, 20)
    assert set(y) == {0, 1, 2, 3}
