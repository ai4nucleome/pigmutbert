import os
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union

import torch
from transformers import AutoModelForSequenceClassification
import transformers
import sklearn
import numpy as np
from datasets import load_dataset
import torch.nn.functional as F
from models.DNABERT2.bert_layers import BertForSequenceClassification as DB2BertForSequenceClassification
from models.MutBERT.modeling_mutbert import RoPEBertForSequenceClassification
from models.GENA_LM.modeling_bert import BertForSequenceClassification as genaForSequenceClassification
from datetime import datetime

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: int = field(default=-1, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    factor: float = field(default=1.0)
    num_labels: int = field(default=2)
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=None)
    max_train_steps: int = field(default=None)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    logging_dir: str = field(default="runs")
    tb_name: str = field(default="run")  # add new arg
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    eval_strategy: str = field(default="steps")
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_matthews_correlation")
    greater_is_better: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


"""
Get the reversed complement of the original DNA sequence.
"""
def get_alter_of_dna_sequence(sequence: str):
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    # return "".join([MAP[c] for c in reversed(sequence)])
    return "".join([MAP[c] for c in sequence])

"""
Transform a dna sequence to k-mer string
"""
def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


"""
Load or generate k-mer string for each DNA sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
"""
def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:        
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)
        
    return kmer


"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    
    # 计算混淆矩阵
    # conf_mat = sklearn.metrics.confusion_matrix(valid_labels, valid_predictions)
    
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        # "confusion_matrix": conf_mat.tolist(),
    }

# from: https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
def preprocess_logits_for_metrics(logits:Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]

    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])

    return torch.argmax(logits, dim=-1)


"""
Compute metrics used for huggingface trainer.
"""
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return calculate_metric_with_sklearn(predictions, labels)


class MyCollateFn:
    def __init__(self, tokenizer, is_oh):
        self.tokenizer = tokenizer
        self.is_oh = is_oh

    def __call__(self, batch):
        batch = self.tokenizer.pad(batch, padding=True, return_tensors="pt")

        if self.is_oh:
            batch["input_ids"] = F.one_hot(batch["input_ids"], num_classes=len(self.tokenizer)).float()

        return batch

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right", 
        use_fast=True,
        trust_remote_code=True,
    )

    # define datasets and data collator
    def preprocess_function(examples):
        if data_args.kmer > 0:
            seqs = [" ".join([seq[i:i+data_args.kmer] for i in range(len(seq) - data_args.kmer + 1)]) for seq in examples["sequence"]]
        else:
            seqs = examples["sequence"]

        tokenized_inputs = tokenizer(seqs, padding=False,
                                 truncation=True,
                                 max_length=tokenizer.model_max_length,
                                 )
        tokenized_inputs['labels'] = examples['label']
        return tokenized_inputs
    
    raw_datasets = load_dataset("csv", data_files={
        "train": os.path.join(data_args.data_path, "train.csv"),
        "validation": os.path.join(data_args.data_path, "dev.csv"), 
        "test": os.path.join(data_args.data_path, "test.csv")
    })

    raw_datasets = raw_datasets.map(preprocess_function,
                                    batched=True,
                                    remove_columns=raw_datasets["train"].column_names,
                                    load_from_cache_file=False,
                                    desc=f"Running tokenizer on dataset")
    raw_datasets.set_format(type="torch")

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]
    test_dataset = raw_datasets["test"]
    
    is_oh = "pig" in model_args.model_name_or_path or "MutBERT" in model_args.model_name_or_path
    is_gena = "GENA" in model_args.model_name_or_path
    data_collator = MyCollateFn(tokenizer=tokenizer, is_oh=is_oh)

    current_time = datetime.now().strftime("%m-%d")
    training_args.logging_dir = f"{training_args.output_dir}/runs/{current_time}_{training_args.tb_name}"
    print(f"Logging directory set to: {training_args.logging_dir}")

    # load model
    if is_oh:
        model = RoPEBertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            rope_scaling={'type': 'dynamic','factor': training_args.factor},
            cache_dir=training_args.cache_dir,
            num_labels=training_args.num_labels,
        )
    elif is_gena:
        model = genaForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=training_args.num_labels,
        )
    elif "DNABERT2" in model_args.model_name_or_path or "GenomicsFM" in model_args.model_name_or_path:
        # dnabert-2 and GenomicsFM
        model = DB2BertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=training_args.num_labels,
        )
    else:
        # NTv1, NTv2
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=training_args.num_labels,
            trust_remote_code=True,
        )


    # configure LoRA
    if model_args.use_lora > 0:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=list(model_args.lora_target_modules.split(",")),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # define trainer
    trainer = transformers.Trainer(model=model,
                                tokenizer=tokenizer,
                                args=training_args,
                                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                                compute_metrics=compute_metrics,
                                train_dataset=train_dataset,
                                eval_dataset=eval_dataset,
                                data_collator=data_collator)
    trainer.train()

    # get the evaluation results from trainer
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    train()
