from pathlib import Path


def get_cache_dir(cache_name: str = "cache"):
    base_dir = Path(__file__).resolve().parents[1]
    cache_dir = base_dir / cache_name
    return cache_dir
