import argparse
import logging
from datetime import datetime

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import optuna.logging

from train import main

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler(f"logs/optimize_{datetime.now().strftime('%Y%m%d')}.log", mode="w"))

optuna.logging.enable_propagation()
optuna.logging.set_verbosity(optuna.logging.INFO)

def objective(trial, base_args):

    # batch_size, num_epochs, max_length
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    num_epochs = trial.suggest_int("num_epochs", 3, 15)
    max_length = trial.suggest_categorical("max_length", [128, 160, 200])

    # learning rate
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    
    # scheduler
    scheduler = trial.suggest_categorical("scheduler", ["linear", "cosine"])

    # focal_loss and corresponding parameters(alpha, gamma)
    focal_loss = trial.suggest_categorical("focal_loss", [True, False])
    if focal_loss:
        alpha = trial.suggest_float("alpha", 0.25, 1.0, step=0.25)
        gamma = trial.suggest_float("gamma", 0.5, 2.0, step=0.5)
    else:
        alpha = None
        gamma = None

    # dropout
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
    # label_smoothing
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2, step=0.05)

    # use_span_pooling, use_attention_pooling, use_entity_markers, use_entity_types
    use_span_pooling = trial.suggest_categorical("use_span_pooling", [True, False])
    use_attention_pooling = trial.suggest_categorical("use_attention_pooling", [True, False])
    use_entity_markers = trial.suggest_categorical("use_entity_markers", [True, False])
    use_entity_types = trial.suggest_categorical("use_entity_types", [True, False])

    args = argparse.Namespace(**vars(base_args))
    args.batch_size = batch_size
    args.num_epochs = num_epochs
    args.max_length = max_length
    args.learning_rate = learning_rate
    args.scheduler = scheduler
    args.focal_loss = focal_loss
    args.alpha = alpha
    args.gamma = gamma
    args.dropout = dropout
    args.label_smoothing = label_smoothing
    args.use_span_pooling = use_span_pooling
    args.use_attention_pooling = use_attention_pooling
    args.use_entity_markers = use_entity_markers
    args.use_entity_types = use_entity_types

    return main(args, trial=trial)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--valid_file", type=str)
    parser.add_argument("--label2id_path", type=str)
    parser.add_argument("--id2label_path", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--scheduler", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--focal_loss", action="store_true")
    parser.add_argument("--label_smoothing", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--use_span_pooling", action="store_true")
    parser.add_argument("--use_attention_pooling", action="store_true")
    parser.add_argument("--use_entity_markers", action="store_true")
    parser.add_argument("--use_entity_types", action="store_true")
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--study_name", type=str)
    parser.add_argument("--n_trials", type=int, default=10)

    args = parser.parse_args()
    return args

def optimize():
    base_args = parse_args()

    def objective_wrapper(trial):
        return objective(trial, base_args)
    
    study = optuna.create_study(direction="maximize", 
                                sampler=TPESampler(seed=42), 
                                pruner=HyperbandPruner(min_resource=1, reduction_factor=3),
                                study_name=base_args.study_name,
                                storage=f"sqlite:///upstage_assignment.db",
                                load_if_exists=True)

    logger.info(f"Optimization Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    study.optimize(objective_wrapper, n_trials=base_args.n_trials)
    logger.info(f"Optimization End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    logger.info("-----------[Optimization Result]---------------")
    logger.info(f"Best Value: {study.best_value}")
    logger.info(f"Best Params: {study.best_params}")
    logger.info("-----------------------------------------------")

    if base_args.save_model:
        final_args = argparse.Namespace(**vars(base_args))
        for k, v in study.best_params.items():
            setattr(final_args, k, v)
        main(final_args, trial=None)

    study.trials_dataframe().to_csv(f"logs/{base_args.study_name}.csv")
    return study

if __name__ == "__main__":
    optimize()