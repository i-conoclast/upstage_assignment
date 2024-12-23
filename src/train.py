import os
import json
from typing import Optional, List, Union, Any, Dict, Tuple
from datetime import datetime
import torch
import torch.nn as nn
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from transformers.trainer_pt_utils import nested_detach
from transformers import (
    Trainer, 
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
    get_scheduler
)
import optuna  

from dataset import RelationDataset
from tools.utils import make_mapping
from model import RelationClassifier
from loss import FocalLoss
from metrics import compute_micro_f1, compute_auprc
from config import (NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, NUM_LABELS, 
                    MODEL_NAME_OR_PATH, MAX_LENGTH, LABEL_SMOOTHING, ALPHA, GAMMA,
                    LABEL2ID_PATH, ID2LABEL_PATH, TRAIN_FILE, VALID_FILE, MODEL_DIR, DROPOUT)

# override compute_loss to use custom loss function
class CustomTrainer(Trainer):
    def __init__(self, focal_loss=False, alpha=1.0, gamma=1.0, label_smoothing=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = focal_loss
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

        if self.focal_loss:
            self.loss_fn = FocalLoss(alpha=self.alpha, gamma=self.gamma, label_smoothing=self.label_smoothing)
        else:
            from torch.nn import CrossEntropyLoss
            self.loss_fn = CrossEntropyLoss(label_smoothing=self.label_smoothing)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)  # (batch_size, num_labels)
        # outputs is logits
        logits = outputs

        loss = self.loss_fn(logits, labels)

        if return_outputs:
            return (loss, outputs)
        return loss
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs # original : outputs[1:] -> fix error
            else:
                loss = None
                if self.use_amp:
                    with autocast():
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)


class OptunaTrialPruningCallback(TrainerCallback):
    def __init__(self, trial, metric_name="micro_f1"):
        self.trial = trial
        self.metric_name = metric_name

    def on_evaluate(self, args, state, control, **kwargs):
        # trainer automatically logs metrics in state.log_history
        # we find the latest 'eval_{metric_name}' if present

        if not hasattr(state, "log_history"):
            return

        # Find the most recent eval record
        current_metric = None
        for record in reversed(state.log_history):
            if f"eval_{self.metric_name}" in record:
                current_metric = record[f"eval_{self.metric_name}"]
                break

        if current_metric is None:
            return

        # step = state.global_step or state.epoch
        step = state.epoch  # or global_step
        step = step if step is not None else state.global_step

        self.trial.report(current_metric, step=step)

        if self.trial.should_prune():
            raise optuna.TrialPruned()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # logits shape: (batch_size, num_labels)
    # labels: (batch_size, )
    probs = torch.softmax(torch.tensor(logits), dim=-1)
    preds = np.argmax(probs, axis=1)

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    
    no_relation = "no_relation"

    micro_f1 = compute_micro_f1(labels, preds, no_relation)
    auprc = compute_auprc(labels, probs)
    return {"micro_f1": micro_f1, "auprc": auprc}


def main(args, trial=None):
    label2id, id2label = make_mapping(args.label2id_path, args.id2label_path)
    num_labels = NUM_LABELS  # from config or len(label2id)

    train_dataset = RelationDataset(
        file_path=args.train_file,
        tokenizer_name=args.model_name_or_path,
        max_length=args.max_length,
        label2id=label2id,
        use_entity_markers=args.use_entity_markers,
        use_entity_types=args.use_entity_types,
        use_span_pooling=args.use_span_pooling,
        inference=False
    )
    valid_dataset = RelationDataset(
        file_path=args.valid_file,
        tokenizer_name=args.model_name_or_path,
        max_length=args.max_length,
        label2id=label2id,
        use_entity_markers=args.use_entity_markers,
        use_entity_types=args.use_entity_types,
        use_span_pooling=args.use_span_pooling,
        inference=False
    )

    model = RelationClassifier(
        model_name=args.model_name_or_path,
        num_labels=num_labels,
        dropout=args.dropout,
        tokenizer_len=len(train_dataset.tokenizer),
        use_span_pooling=args.use_span_pooling,
        use_attention_pooling=args.use_attention_pooling
    )

    folder_num = [int(folder.split("_")[-1]) for folder in os.listdir(args.model_dir) if folder.startswith("best_model_") and os.path.isdir(folder)] + [0]
    folder_name = f"best_model_{datetime.now().strftime('%Y%m%d')}_{max(folder_num) + 1}"
    os.makedirs(os.path.join(args.model_dir, folder_name), exist_ok=True)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.model_dir, folder_name),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        lr_scheduler_type=args.scheduler,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",  # compute_metrics() returns micro_f1
        greater_is_better=True,            # F1 => higher better
        logging_dir=os.path.join(args.model_dir, "logs"),
        report_to="none",      # or "tensorboard"
    )
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=0.0
    )

    pruning_callback = OptunaTrialPruningCallback(trial, metric_name="micro_f1")

    if trial is None:
        callbacks = [early_stopping]
    else:
        callbacks = [early_stopping, pruning_callback]
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        focal_loss=args.focal_loss,
        alpha=args.alpha,
        gamma=args.gamma,
        label_smoothing=args.label_smoothing,
        callbacks=callbacks
    )

    trainer.train()

    #if args.save_model:
       # trainer.save_model(os.path.join(args.model_dir, folder_name))

    with open(os.path.join(args.model_dir, folder_name, "best_model_config.json"), "w") as f:
        args_dict = vars(args)
        args_dict["micro_f1"] = trainer.state.best_metric
        args_dict["best_checkpoint_path"] = trainer.state.best_model_checkpoint
        json.dump(args_dict, f, ensure_ascii=False, indent=2)

    print("Best model is saved at:", trainer.state.best_model_checkpoint)

    return trainer.state.best_metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=MODEL_NAME_OR_PATH)
    parser.add_argument("--train_file", type=str, default=TRAIN_FILE)
    parser.add_argument("--valid_file", type=str, default=VALID_FILE)
    parser.add_argument("--label2id_path", type=str, default=LABEL2ID_PATH)
    parser.add_argument("--id2label_path", type=str, default=ID2LABEL_PATH)
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--scheduler", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"], default="linear")
    parser.add_argument("--focal_loss", action="store_true")
    parser.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--use_span_pooling", action="store_true")
    parser.add_argument("--use_attention_pooling", action="store_true")
    parser.add_argument("--use_entity_markers", action="store_true")
    parser.add_argument("--use_entity_types", action="store_true")
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    args = parser.parse_args()
    main(args)
