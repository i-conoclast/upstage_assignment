import torch
import torch.nn as nn
from transformers import AutoModel

class RelationClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1, tokenizer_len: int = None):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        # due to span pooling, hidden size is doubled : [E1_vec; E2_vec] concat -> hidden_size * 2
        self.classifier = nn.Linear(self.model.config.hidden_size * 2, num_labels)
        if tokenizer_len:
            self.model.resize_token_embeddings(tokenizer_len)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                labels: torch.Tensor = None, e1_start_idx: torch.Tensor = None, 
                e1_end_idx: torch.Tensor = None, e2_start_idx: torch.Tensor = None, 
                e2_end_idx: torch.Tensor = None) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        if self.use_span_pooling:
            # span pooling
            # E1 pooling
            e1_pooled = []
            for i, hs in enumerate(hidden_states):
                start = e1_start_idx[i].item()
                end = e1_end_idx[i].item()
                # to handle case where area is not correct
                if end >= start:
                    span_vec = hs[start:end+1].mean(dim=0)
                else:
                    # fallback: CLS or avg but here use CLS
                    span_vec = hs[0]
                e1_pooled.append(span_vec)
            e1_pooled = torch.stack(e1_pooled, dim=0)

            # E2 pooling
            e2_pooled = []
            for i, hs in enumerate(hidden_states):
                start = e2_start_idx[i].item()
                end = e2_end_idx[i].item()
                # to handle case where area is not correct
                if end >= start:
                    span_vec = hs[start:end+1].mean(dim=0)
                else:
                    # fallback: CLS or avg but here use CLS
                    span_vec = hs[0]
                e2_pooled.append(span_vec)
            e2_pooled = torch.stack(e2_pooled, dim=0)

            # concat
            concat_vec = torch.cat([e1_pooled, e2_pooled], dim=-1)
            concat_vec = self.dropout(concat_vec)
            logits = self.classifier(concat_vec)
        else:
            cls_output = outputs.last_hidden_state[:, 0, :]
            cls_output = self.dropout(cls_output)
            logits = self.classifier(cls_output)
        return logits
