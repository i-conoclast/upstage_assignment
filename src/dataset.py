import ast
import pandas as pd
from torch.util.data import Dataset
from transformers import AutoTokenizer

from config import (E1_MARKER, E1_MARKER_CLOSE, E2_MARKER, E2_MARKER_CLOSE,
                    E1_MARKER_PREFIX, E1_MARKER_CLOSE_PREFIX, E2_MARKER_PREFIX, E2_MARKER_CLOSE_PREFIX)

class RelationDataset(Dataset):
    def __init__(self, file_path: str, tokenizer_name: str, max_length: int,
                 label2id: dict, use_entity_markers: bool = True,
                 use_entity_types: bool = False, use_span_pooling: bool = False, 
                 inference: bool = False):
        
        self.data = pd.read_csv(file_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.label2id = label2id
        self.use_entity_markers = use_entity_markers
        self.use_entity_types = use_entity_types
        self.use_span_pooling = use_span_pooling
        self.inference = inference

        self.data["subject_entity"] = self.data["subject_entity"].apply(ast.literal_eval)
        self.data["object_entity"] = self.data["object_entity"].apply(ast.literal_eval)
        if self.use_span_pooling:
            self.tokenizer.add_tokens([E1_MARKER, E1_MARKER_CLOSE, E2_MARKER, E2_MARKER_CLOSE])

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        item = self.data.iloc[idx]

        sentence = item["sentence"]
        sub = item["subject_entity"]
        obj = item["object_entity"]
        idx = item["id"]
        
        if self.use_entity_markers:
            sentence = self.insert_entity_markers(sentence, sub, obj)

        encoding = self.tokenizer(sentence, 
                                  truncation=True, 
                                  padding="max_length", 
                                  max_length=self.max_length,
                                  padding="max_length",
                                  return_tensors="pt")
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        e1_start_idx, e1_end_idx = None, None
        e2_start_idx, e2_end_idx = None, None
        if self.use_span_pooling:
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            e1_start_idx, e1_end_idx = self.find_span(tokens, E1_MARKER_PREFIX, E1_MARKER_CLOSE_PREFIX)
            e2_start_idx, e2_end_idx = self.find_span(tokens, E2_MARKER_PREFIX, E2_MARKER_CLOSE_PREFIX)

        if not self.inference:
            label = item["label"]
            label_id = self.label2id[label]

            return {"id": idx, 
                    "input_ids": input_ids, 
                    "attention_mask": attention_mask, 
                    "label": label_id,
                    "e1_start_idx": e1_start_idx, 
                    "e1_end_idx": e1_end_idx, 
                    "e2_start_idx": e2_start_idx, 
                    "e2_end_idx": e2_end_idx}
        else:
            return {"id": idx, 
                    "input_ids": input_ids, 
                    "attention_mask": attention_mask, 
                    "e1_start_idx": e1_start_idx, 
                    "e1_end_idx": e1_end_idx, 
                    "e2_start_idx": e2_start_idx, 
                    "e2_end_idx": e2_end_idx}
        
    def insert_entity_markers(self, sentence: str, sub: dict, obj: dict) -> str:
        s_start, s_end = sub["start_idx"], sub["end_idx"]
        o_start, o_end = obj["start_idx"], obj["end_idx"]

        if self.use_entity_types:
            s_type = sub["type"]
            o_type = obj["type"]
            s_open_marker = f"[E1-{s_type}]"
            s_close_marker = f"[/E1-{s_type}]"
            o_open_marker = f"[E2-{o_type}]"
            o_close_marker = f"[/E2-{o_type}]"
        else:
            s_open_marker = "[E1]"
            s_close_marker = "[/E1]"
            o_open_marker = "[E2]"
            o_close_marker = "[/E2]"
        
        if s_start < o_start:
            # insert subject entity marker first
            sentence = sentence[:s_start] + s_open_marker + sentence[s_start:s_end+1] + s_close_marker + sentence[s_end+1:]
            # changed sentence length, so need to update object entity marker start index, end index
            o_start += len(s_open_marker) + len(s_close_marker)
            o_end += len(s_open_marker) + len(s_close_marker)
            # insert object entity marker next
            sentence = sentence[:o_start] + o_open_marker + sentence[o_start:o_end+1] + o_close_marker + sentence[o_end+1:]
        else:
            # insert object entity marker first
            sentence = sentence[:o_start] + o_open_marker + sentence[o_start:o_end+1] + o_close_marker + sentence[o_end+1:]
            # changed sentence length, so need to update subject entity marker start index, end index
            s_start += len(o_open_marker) + len(o_close_marker)
            s_end += len(o_open_marker) + len(o_close_marker)
            # insert subject entity marker next
            sentence = sentence[:s_start] + s_open_marker + sentence[s_start:s_end+1] + s_close_marker + sentence[s_end+1:]

        return sentence

    def find_span(self, tokens: list, 
                  marker_open_prefix: str = E1_MARKER_PREFIX, 
                  marker_close_prefix: str = E1_MARKER_CLOSE_PREFIX) -> tuple:
        open_idx = None
        close_idx = None

        for i, token in enumerate(tokens):
            if token.startswith(marker_open_prefix):
                open_idx = i
            elif token.startswith(marker_close_prefix):
                close_idx = i
        
        # except marker open prefix and marker close prefix, find span
        if open_idx is not None and close_idx is not None and open_idx < close_idx:
            return open_idx+1, close_idx-1
        else:
            return None, None
        




