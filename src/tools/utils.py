import pickle

def make_mapping(dict_1_path: str, dict_2_path: str) -> dict:
    with open(dict_1_path, "rb") as f:
        label2id = pickle.load(f)
    with open(dict_2_path, "rb") as f:
        id2label = pickle.load(f)
    return label2id, id2label