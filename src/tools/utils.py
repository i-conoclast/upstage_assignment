import pickle
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

def make_mapping(dict_1_path: str, dict_2_path: str) -> dict:
    with open(PROJECT_DIR / dict_1_path, "rb") as f:
        label2id = pickle.load(f)
    with open(PROJECT_DIR / dict_2_path, "rb") as f:
        id2label = pickle.load(f)
    return label2id, id2label