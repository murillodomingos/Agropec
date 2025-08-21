
from pathlib import Path


def get_data_dir():
    return Path("data")

def get_raw_dir():
    return get_data_dir() / "00--raw"

def get_clean_dir():
    return get_clean_dir() / "01--clean"