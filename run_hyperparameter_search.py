import argparse
import transformers
import boolq
import data_utils
import finetuning_utils
import json
import pandas as pd
from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest.bayesopt import BayesOptSearch
from transformers import RobertaForSequenceClassification

from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast
from transformers import TrainingArguments, Trainer

parser = argparse.ArgumentParser(
    description="Run a hyperparameter search for finetuning a RoBERTa model on the BoolQ dataset."
)
parser.add_argument(
    "data_dir",
    type=str,
    help="Directory containing the BoolQ dataset. Can be downloaded from https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip.",
)

args = parser.parse_args()


train_df = pd.read_json(f"{args.data_dir}", lines=True, orient="records")
val_df, test_df = train_test_split(
    pd.read_json(f"{args.data_dir}", lines=True, orient="records"),
    test_size=0.5,
)

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
train_data = boolq.BoolQDataset(train_df, tokenizer)
val_data = boolq.BoolQDataset(val_df, tokenizer)
test_data = boolq.BoolQDataset(test_df, tokenizer)

args = TrainingArguments(
    output_dir="/content",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    seed=0,
    load_best_model_at_end=True,
    learning_rate = 1e-5,
    do_train=True,
    do_eval=True,

)

trainer = Trainer(
    model_init = finetuning_utils.model_init,
    args = args,
    train_dataset = train_data,
    eval_dataset = val_data,
    #data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer),
    compute_metrics = finetuning_utils.compute_metrics    
)

def my_hp_space(trial):
    return {
        "learning_rate": tune.loguniform(1e-5, 5e-5)
    }

reporter = CLIReporter(
    parameter_columns={
        "weight_decay": "w_decay",
        "learning_rate": "lr",
        "per_device_train_batch_size": "train_bs/gpu",
        "num_train_epochs": "num_epochs"
    },
    metric_columns=[
        "eval_s_accuracy", "eval_loss", "eval_s_f1", "steps"
    ])
def my_objective(metrics):
    result_to_optimize = metrics['eval_loss']
    return result_to_optimize

hpsearch = trainer.hyperparameter_search(
    hp_space = my_hp_space,
    n_trials = 5,
    direction = 'maximize',
    backend="ray",
    search_alg = BayesOptSearch(metric="mean_loss"),
    mode="min",
    progress_reporter = reporter,
    log_to_file = True,
    local_dir = '/scratch/ibd214'
)
