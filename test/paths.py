import os
import shutil

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
TMP_DIR = os.path.join(os.path.dirname(__file__), "tmp")


def clean_tmp_dir():
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
