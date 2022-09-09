import os


def ensure_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
