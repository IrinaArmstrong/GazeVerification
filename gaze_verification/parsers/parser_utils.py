import os

def is_dir_empty(dir_path):
    return not next(os.scandir(dir_path), None)