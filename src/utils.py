import os
import sys
import joblib

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """
    Saves any Python object (model, scaler, etc.) do disk using joblib
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        joblib.dump(obj, file_path)
        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Loads a previously saved Python object (model, scaler, etc.) from disk.
    """

    try:
        return joblib.load(file_path)

    except Exception as e:
        raise CustomException(e, sys)
