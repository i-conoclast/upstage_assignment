import os
import json
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from dataset import RelationDataset
from tools.utils import make_mapping
from model import RelationClassifier
from loss import FocalLoss
from metrics import compute_micro_f1, compute_auprc
from config import (NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, NUM_LABELS, 
                    MODEL_NAME_OR_PATH, MAX_LENGTH, LABEL_SMOOTHING, ALPHA, GAMMA,
                    LABEL2ID_PATH, ID2LABEL_PATH, TRAIN_FILE, VALID_FILE, MODEL_DIR, DROPOUT)

import argparse
from tqdm import tqdm

def main(args):
    label2id, id2label = make_mapping(args.label2id_path, args.id2label_path)
    num_labels = NUM_LABELS

    train_dataset = RelationDataset(args.train_file, args.model_name_or_path, args.max_length, label2id, 
                                    use_entity_markers=args.use_entity_markers, use_entity_types=args.use_entity_types,
                                    use_span_pooling=args.use_span_pooling, inference=False)
    valid_dataset = RelationDataset(args.valid_file, args.model_name_or_path, args.max_length, label2id, 
                                    use_entity_markers=args.use_entity_markers, use_entity_types=args.use_entity_types,
                                    use_span_pooling=args.use_span_pooling, inference=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    model = RelationClassifier(args.model_name_or_path, num_labels, args.dropout, 
                               len(train_dataset.tokenizer), args.use_span_pooling)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    num_training_steps = len(train_loader) * args.num_epochs
    if args.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(num_training_steps*0.1), num_training_steps=num_training_steps)
    elif args.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(num_training_steps*0.1), num_training_steps=num_training_steps)

    if args.focal_loss:
        loss_fn = FocalLoss(alpha=args.alpha, gamma=args.gamma, label_smoothing=args.label_smoothing)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_f1 = 0.0
    best_auprc = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")
    else:
        device = torch.device("mps")
    
    model.to(device)

    # training
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            batch.pop("id")
            labels = batch.pop("label")

            optimizer.zero_grad()
            logits = model(**batch)
            train_loss_val = loss_fn(logits, labels)
            train_loss_val.backward()
            optimizer.step()
            scheduler.step()
            train_loss += train_loss_val.item()

        avg_train_loss = train_loss / len(train_loader)

        # validation
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        for batch in tqdm(valid_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            batch.pop("id")
            labels = batch.pop("label")

            logits = model(**batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())


        # calculate metrics
        no_relation = "no_relation"
        micro_f1 = compute_micro_f1(all_labels, all_preds, no_relation)
        auprc = compute_auprc(all_labels, all_probs)

        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} - Valid Micro F1: {micro_f1:.4f} - Valid AUPRC: {auprc:.4f}")

        if micro_f1 > best_f1:
            best_f1 = micro_f1
            if args.save_model:
                model_base_name = f"best_model_{datetime.now().strftime('%Y%m%d')}"
                model_num = len([f for f in os.listdir(args.output_dir) if f.startswith(model_base_name)])
                model_name = f"{model_base_name}_{model_num+1}"
                with open(os.path.join(args.output_dir, model_name+".json"), "w") as f:
                    json.dump(args.__dict__, f, ensure_ascii=False)
                torch.save(model.state_dict(), os.path.join(args.output_dir, model_name+".pth"))
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=MODEL_NAME_OR_PATH)
    parser.add_argument("--train_file", type=str, default=TRAIN_FILE)
    parser.add_argument("--valid_file", type=str, default=VALID_FILE)
    parser.add_argument("--label2id_path", type=str, default=LABEL2ID_PATH)
    parser.add_argument("--id2label_path", type=str, default=ID2LABEL_PATH)
    parser.add_argument("--output_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--scheduler", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--focal_loss", action="store_true")
    parser.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--use_span_pooling", action="store_true")
    parser.add_argument("--use_entity_markers", action="store_true")
    parser.add_argument("--use_entity_types", action="store_true")
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    args = parser.parse_args()

    main(args)