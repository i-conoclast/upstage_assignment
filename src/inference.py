import os
import json
import argparse
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model import RelationClassifier
from dataset import RelationDataset
from tools.utils import make_mapping
from config import MODEL_DIR, OUTPUT_DIR, TEST_FILE


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")

    model_config_file = os.path.join(args.model_dir, args.model_file + ".json")
    with open(model_config_file, "r") as f:
        model_config = json.load(f)

    label2id, id2label = make_mapping(model_config["label2id_path"], model_config["id2label_path"])
    test_dataset = RelationDataset(args.test_file, model_config["model_name_or_path"], model_config["max_length"], 
                                    label2id, use_entity_markers=model_config["use_entity_markers"], 
                                    use_entity_types=model_config["use_entity_types"], use_span_pooling=model_config["use_span_pooling"], 
                                    inference=True)
    test_loader = DataLoader(test_dataset, batch_size=model_config["batch_size"], shuffle=False)
    
    model = RelationClassifier(model_config["model_name_or_path"], len(label2id), model_config["dropout"], 
                               len(test_dataset.tokenizer), model_config["use_span_pooling"])
    model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_file + ".pth")))
    model.to(device)
    model.eval()

    all_preds = []
    all_probs = []
    all_ids = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            batch.pop("id")
            
            logits = model(**batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            preds = preds.cpu().numpy().tolist()
            preds = [id2label[pred] for pred in preds]
            
            all_preds.extend(preds)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_ids.extend(batch["id"].cpu().numpy().tolist())
   
    results = pd.DataFrame({"id": all_ids, "pred_label": all_preds, "probs": all_probs})
    results["id"] = results["id"].astype(int)
    results.sort_values(by="id", inplace=True)
    results.to_csv(os.path.join(args.output_dir, args.model_file + ".csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--model_file", type=str)
    parser.add_argument("--test_file", type=str, default=TEST_FILE)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--use_cuda", action="store_true")
    args = parser.parse_args()
    main(args)