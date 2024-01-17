from pathlib import Path
import yaml
from src.logger import logger
import os
import pickle

def read_yaml(file_path: Path):
    
    try:
        with open(file_path) as f:
            file_content = yaml.safe_load(f)
            logger.info(f"yaml file: {f} loaded successfully")
            return file_content
    except Exception as e:
        raise e
    

def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise Exception(e)