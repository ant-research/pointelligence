# Exclude standalone benchmark scripts from pytest collection.
# These are meant to be run directly (python script.py), not via pytest.
collect_ignore_glob = [
    "backbone/*",
    "single_conv/*",
]
