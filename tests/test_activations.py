import numpy as np

from src.activations import cache_key, cached_compute


def test_cache_key_is_stable():
    k1 = cache_key("a", 1, "prompt text")
    k2 = cache_key("a", 1, "prompt text")
    assert k1 == k2


def test_cache_key_differs_for_different_inputs():
    assert cache_key("a", 1, "x") != cache_key("a", 1, "y")
    assert cache_key("a", 1, "x") != cache_key("a", 2, "x")


def test_cached_compute_calls_compute_fn_once(tmp_path):
    calls = {"count": 0}

    def compute():
        calls["count"] += 1
        return np.array([1.0, 2.0, 3.0])

    key = cache_key("test", "only-once")

    first = cached_compute(key, compute, cache_dir=tmp_path)
    second = cached_compute(key, compute, cache_dir=tmp_path)

    assert calls["count"] == 1
    np.testing.assert_array_equal(first, second)


def test_cached_compute_writes_to_disk(tmp_path):
    key = cache_key("test", "writes-to-disk")
    cached_compute(key, lambda: np.array([1.0]), cache_dir=tmp_path)
    assert (tmp_path / f"{key}.npy").exists()


def test_cached_compute_different_keys_dont_collide(tmp_path):
    results = {}

    def compute_a():
        return np.array([1.0])

    def compute_b():
        return np.array([2.0])

    key_a = cache_key("a")
    key_b = cache_key("b")

    results["a"] = cached_compute(key_a, compute_a, cache_dir=tmp_path)
    results["b"] = cached_compute(key_b, compute_b, cache_dir=tmp_path)

    assert results["a"][0] == 1.0
    assert results["b"][0] == 2.0
