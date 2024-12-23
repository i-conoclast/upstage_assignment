import os
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import safetensors
from model import RelationClassifier
from dataset import RelationDataset
from tools.utils import make_mapping
from config import OUTPUT_DIR, TEST_FILE, MODEL_DIR, LABEL2ID_PATH, ID2LABEL_PATH

def ensemble_hard(label_arrays):
    from scipy.stats import mode
    arr = np.stack(label_arrays, axis=0) # (num_models, num_samples)
    majority, _ = mode(arr, axis=0)
    final_labels = majority.squeeze(0)
    return final_labels, None # probs are not available for hard voting

def ensemble_soft(prob_arrays):
    arr = np.stack(prob_arrays, axis=0) # (num_models, num_samples, num_labels)
    avg_probs = np.mean(arr, axis=0) # (num_samples, num_labels)
    final_labels = np.argmax(avg_probs, axis=1)
    return final_labels, avg_probs

def ensemble_logit(logit_arrays):
    arr = np.stack(logit_arrays, axis=0) # (num_models, num_samples, num_labels)
    avg_logits = np.mean(arr, axis=0) # (num_samples, num_labels)
    exp_s = np.exp(avg_logits - np.max(avg_logits, axis=1, keepdims=True))
    final_probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
    final_labels = np.argmax(final_probs, axis=1)
    return final_labels, final_probs

def ensemble_weighted_logit(logit_arrays, weights):
    arr = np.stack(logit_arrays, axis=0) # (num_models, num_samples, num_labels)
    w = np.array(weights).reshape(-1, 1, 1) # (num_models, 1, 1)
    weighted = arr * w
    sum_logits = np.sum(weighted, axis=0) # (num_samples, num_labels)
    exp_s = np.exp(sum_logits - np.max(sum_logits, axis=1, keepdims=True))
    final_probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
    final_labels = np.argmax(final_probs, axis=1)
    return final_labels, final_probs

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")

    model_files = args.model_files.split(",")
    ensemble_mode = args.ensemble_mode.lower()

    # weighted
    manual_weights = None
    if ensemble_mode == "weighted" and args.weights:
        manual_weights = list(map(float, args.weights.split(",")))
        if len(manual_weights) != len(model_files):
            raise ValueError(f"Number of weights ({len(manual_weights)}) does not match the number of models ({len(model_files)})")

    label2id, id2label = make_mapping(args.label2id_path, args.id2label_path)
    all_ids = None
    num_samples = 0

    # for weighted ensemble
    f1_list = []

    # ensemble data
    all_labels_dict = {}
    all_probs_dict = {}
    all_logits_dict = {}

    for i, mf in enumerate(model_files):
        config_path = os.path.join(args.model_dir, 
                                  mf, 
                                  "best_model_config.json")
        with open(config_path, "r") as f:
            conf = json.load(f)
        model_path = os.path.join(conf["best_checkpoint_path"],  
                                  "model.safetensors")

        num_labels = len(label2id)

        f1 = conf.get("micro_f1")
        f1_list.append(f1)
        # load data
        ds = RelationDataset(args.test_file, 
                             conf["model_name_or_path"], 
                             conf["max_length"], 
                             label2id, 
                             use_entity_markers=conf.get("use_entity_markers", False), 
                             use_entity_types=conf.get("use_entity_types", False), 
                             use_span_pooling=conf.get("use_span_pooling", False), 
                             inference=True)
        dl = DataLoader(ds, batch_size=conf["batch_size"], shuffle=False)

        model = RelationClassifier(conf["model_name_or_path"], 
                                   num_labels, 
                                   conf["dropout"], 
                                   len(ds.tokenizer), 
                                   conf.get("use_span_pooling", False), 
                                   conf.get("use_attention_pooling", False))
        model.load_state_dict(safetensors.torch.load_file(model_path, map_location=device))
        model.eval().to(device)

        sample_ids = []
        model_logits = []
        model_labels = []
        model_probs = []

        with torch.no_grad():
            for batch in tqdm(dl, desc=f"Inference {mf}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                ids = batch.pop("id")

                logits = model(**batch)
                np_logits = logits.cpu().numpy()

                if ensemble_mode == "hard":
                    preds = np.argmax(np_logits, axis=1)
                    model_labels.append(preds)
                elif ensemble_mode == "soft":
                    exp_s = np.exp(np_logits - np.max(np_logits, axis=1, keepdims=True))
                    probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
                    model_probs.append(probs)
                else:
                    # logit or weighted logit
                    model_logits.append(np_logits)
                
                for idx in ids:
                    sample_ids.append(int(idx.item()))

        sample_ids = np.array(sample_ids)
        idx_order = np.argsort(sample_ids)
        sample_ids = sample_ids[idx_order]

        if ensemble_mode == "hard":
            concat_labels = np.concatenate(model_labels, axis=0)[idx_order]
            all_labels_dict[mf] = concat_labels
            num_samples = concat_labels.shape[0]
        elif ensemble_mode == "soft":
            concat_probs = np.concatenate(model_probs, axis=0)[idx_order]
            all_probs_dict[mf] = concat_probs
            num_samples = concat_probs.shape[0]
        else:
            concat_logits = np.concatenate(model_logits, axis=0)[idx_order]
            all_logits_dict[mf] = concat_logits
            num_samples = concat_logits.shape[0]

        if all_ids is None:
            all_ids = sample_ids
        else:
            assert np.array_equal(all_ids, sample_ids), "Sample IDs are not consistent across models"

    # ensemble
    final_labels = None
    final_probs = None

    if ensemble_mode == "hard":
        label_arrays = [all_labels_dict[mf] for mf in model_files]
        final_labels, final_probs = ensemble_hard(label_arrays)
    elif ensemble_mode == "soft":
        probs_arrays = [all_probs_dict[mf] for mf in model_files]
        final_labels, final_probs = ensemble_soft(probs_arrays)
    elif ensemble_mode in ["logit", "weighted"]:
        logits_arrays = [all_logits_dict[mf] for mf in model_files]
        if ensemble_mode == "logit":
            final_labels, final_probs = ensemble_logit(logits_arrays)
        else:
            if manual_weights:
                final_labels, final_probs = ensemble_weighted_logit(logits_arrays, manual_weights)
            else:
                sum_f1 = sum(f1_list)
                auto_weights = [f1 / sum_f1 for f1 in f1_list]
                final_labels, final_probs = ensemble_weighted_logit(logits_arrays, auto_weights)
    else:
        raise ValueError(f"Invalid ensemble mode: {ensemble_mode}")
    
    str_labels = [id2label[label] for label in final_labels]

    data_dict = {
        "id": all_ids,
        "pred_label": str_labels,
    }

    if final_probs is not None:
        prob_list_col = []
        for row in final_probs:
            prob_list_col.append(row.tolist())
        data_dict["probs"] = prob_list_col
    
    out_df = pd.DataFrame(data_dict)
    out_df["id"] = out_df["id"].astype(int)
    out_df.sort_values(by="id", inplace=True)

    out_name = f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    out_df.to_csv(os.path.join(args.output_dir, out_name), index=False)

    ensemble_cfg = args.__dict__
    with open(os.path.join(args.output_dir, out_name.replace(".csv", ".json")), "w") as f:
        json.dump(ensemble_cfg, f, ensure_ascii=False)

    print(f"[Ensemble] saved => {os.path.join(args.output_dir, out_name)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--model_files", type=str)
    parser.add_argument("--ensemble_mode", choices=["hard","soft","logit","weighted"], default="logit")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--test_file", type=str, default=TEST_FILE)
    parser.add_argument("--label2id_path", type=str, default=LABEL2ID_PATH)
    parser.add_argument("--id2label_path", type=str, default=ID2LABEL_PATH)
    parser.add_argument("--use_cuda", action="store_true")
    args = parser.parse_args()
    main(args)  