import torch
import torch.nn as nn
from transformers import AutoModel

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(hidden_size))
    
    def forward(self, span_hidden_states: torch.Tensor) -> torch.Tensor:
        # span_hidden_states: [span_length, hidden_size]
        # return: [hidden_size]

        # attention score : dot product between span_hidden_states(token vectors) and w
        # [span_length, hidden_size] · [hidden_size] -> [span_length]
        scores = torch.matmul(span_hidden_states, self.w)

        # softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=0) # [span_length]
        
        # weighted sum of span_hidden_states
        # (span_length, ) x (span_length, hidden_size) -> sum after broadcasting
        pooled = torch.sum(attn_weights.unsqueeze(-1) * span_hidden_states, dim=0)
        return pooled

class RelationClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, 
                 dropout: float = 0.1, tokenizer_len: int = None,
                 use_span_pooling: bool = False,
                 use_attention_pooling: bool = False):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.use_span_pooling = use_span_pooling
        self.use_attention_pooling = use_attention_pooling
        if self.use_span_pooling:
            # due to span pooling, hidden size is doubled : [E1_vec; E2_vec] concat -> hidden_size * 2
            self.classifier = nn.Linear(self.model.config.hidden_size * 2, num_labels)
        else:
            self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
        
        if self.use_attention_pooling:
            self.e1_attention_pooling = AttentionPooling(self.model.config.hidden_size)
            self.e2_attention_pooling = AttentionPooling(self.model.config.hidden_size)

        if tokenizer_len:
            self.model.resize_token_embeddings(tokenizer_len)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                e1_start_idx: torch.Tensor = None, e1_end_idx: torch.Tensor = None, 
                e2_start_idx: torch.Tensor = None, e2_end_idx: torch.Tensor = None,
                labels=None) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state # [batch_size, seq_len, hidden_size]

        if self.use_span_pooling:
            # span pooling
            # E1 pooling
            e1_pooled = []
            for i, hs in enumerate(hidden_states): # hs : [seq_len, hidden_size]
                start = e1_start_idx[i].item()
                end = e1_end_idx[i].item()
                # to handle case where area is not correct
                if end >= start:
                    span_hs = hs[start:end+1] # [span_length, hidden_size]
                    if not self.use_attention_pooling:
                        span_vec = span_hs.mean(dim=0)
                    else:
                        span_vec = self.e1_attention_pooling(span_hs)
                else:
                    # fallback: CLS or avg but here use CLS
                    span_vec = hs[0]
                e1_pooled.append(span_vec)
            e1_pooled = torch.stack(e1_pooled, dim=0) # [batch_size, hidden_size]

            # E2 pooling
            e2_pooled = []
            for i, hs in enumerate(hidden_states):
                start = e2_start_idx[i].item()
                end = e2_end_idx[i].item()
                # to handle case where area is not correct
                if end >= start:
                    span_hs = hs[start:end+1] # [span_length, hidden_size]
                    if not self.use_attention_pooling:
                        span_vec = span_hs.mean(dim=0)
                    else:
                        span_vec = self.e2_attention_pooling(span_hs)
                else:
                    # fallback: CLS or avg but here use CLS
                    span_vec = hs[0]
                e2_pooled.append(span_vec)
            e2_pooled = torch.stack(e2_pooled, dim=0) # [batch_size, hidden_size]

            # concat
            concat_vec = torch.cat([e1_pooled, e2_pooled], dim=-1) # [batch_size, hidden_size * 2]
            concat_vec = self.dropout(concat_vec)
            logits = self.classifier(concat_vec) # [batch_size, num_labels]
        else:
            # use cls token
            cls_output = outputs.last_hidden_state[:, 0, :] # [batch_size, hidden_size]
            cls_output = self.dropout(cls_output)
            logits = self.classifier(cls_output) # [batch_size, num_labels]
        
        return logits
