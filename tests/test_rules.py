from pcacluster.rules import d_from_K


def test_d_from_K():
    assert d_from_K(10) == 10
    assert d_from_K(10, minus_one=True) == 9
