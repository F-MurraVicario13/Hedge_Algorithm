from backtester.data.cache import DiskCache


def test_cache_miss_then_hit(tmp_path):
    cache = DiskCache(cache_dir=tmp_path, min_interval=0.0)
    calls = []

    def fetch():
        calls.append(1)
        return {"value": 42}

    first = cache.get_json("key-a", fetch)
    second = cache.get_json("key-a", fetch)

    assert first == {"value": 42}
    assert second == {"value": 42}
    assert len(calls) == 1  # second call must be served from disk, not re-fetched


def test_cache_persists_across_instances(tmp_path):
    calls = []

    def fetch():
        calls.append(1)
        return {"value": "persisted"}

    DiskCache(cache_dir=tmp_path, min_interval=0.0).get_json("key-b", fetch)

    # a fresh DiskCache instance pointed at the same directory should hit disk, not fetch again
    second_cache = DiskCache(cache_dir=tmp_path, min_interval=0.0)
    result = second_cache.get_json("key-b", fetch)

    assert result == {"value": "persisted"}
    assert len(calls) == 1


def test_different_keys_do_not_collide(tmp_path):
    cache = DiskCache(cache_dir=tmp_path, min_interval=0.0)
    cache.get_json("key-1", lambda: {"v": 1})
    cache.get_json("key-2", lambda: {"v": 2})

    assert cache.get_json("key-1", lambda: (_ for _ in ()).throw(AssertionError("should be cached"))) == {"v": 1}
    assert cache.get_json("key-2", lambda: (_ for _ in ()).throw(AssertionError("should be cached"))) == {"v": 2}
