import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, auc

from config import LABELS_LIST

def compute_micro_f1(labels, preds, no_relation: str = "no_relation", labels_list: list = LABELS_LIST) -> float:
    # exclude no_relation
    no_relation_label_idx = labels_list.index(no_relation)
    labels_indices = list(range(len(labels_list)))
    labels_indices.remove(no_relation_label_idx)
    
    return f1_score(labels, preds, average="micro", zero_division=0, labels=labels_indices)

def compute_auprc(labels, probs):
    labels = np.array(labels)
    probs = np.array(probs)
    num_classes = probs.shape[1]

    # one-hot encoding
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    
    # compute per-class PR curve
    auprcs = np.zeros((num_classes, ))
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(one_hot[:, i], probs[:, i])
        auprcs[i] = auc(recall, precision)
    
    return np.mean(auprcs)
    