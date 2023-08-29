import warnings

warnings.filterwarnings("ignore")

import os
import math
import unicodedata

from typing import Sequence

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    AutoConfig,
)
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from dataclasses import dataclass, field
from huggingface_hub import login

from transformers import pipeline


login(token="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", add_to_git_credential=True)

# https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
SESSION_NAME = "gpt1_re-pretrain"
CONFIG = {
    # work 2 do
    "do_train": True,
    "do_eval": True,
    # model hyperparameters
    "model_name_or_path": "openai-gpt",
    "fp16": True if torch.cuda.is_available() else False,
    "torch_compile": True,
    # training hyperparameters
    "mini_batch_size": 32,
    "optim": "adamw_hf",
    "triton": True,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 2,
    "warmup_ratio": 0.18,
    "max_steps": 500_000,
    "num_train_epochs": 1024,
    "save_total_limit": 5,
    # dataset
    "dataset_name": "thanhduycao/viet_news_1",
    "dataset_name_config": "default",
    "text_column_name": "text",
    "block_size": 256,
    # eval hyperparameters
    "evaluation_strategy": "steps",
    "eval_steps": 50_000,
    # directories
    "output_dir": "/kaggle/output/",
    "save_strategy": "steps",
    "save_steps": 50_000,
    # other parameters and hub
    "seed": 42,
    "push_to_hub": True,
    "hub_model_id": SESSION_NAME,
    "hub_strategy": "all_checkpoints",
    # load best model at end for inference
    "load_best_model_at_end": True,
    # logging
    "logging_first_step": True,
    "logging_steps": 500,
    "report_to": "wandb",
    "run_name": SESSION_NAME,
    # random seed
    "seed": 42,
    "data_seed": 42,
}

# keep valid arguments
valid_args = {
    k: v
    for k, v in CONFIG.items()
    if k in TrainingArguments.__init__.__code__.co_varnames
}

training_args = TrainingArguments(**valid_args)

for key, value in valid_args.items():
    print(f"{key}: {value}")

tokenizer = AutoTokenizer.from_pretrained(
    CONFIG["model_name_or_path"],
    padding_side="left",
    use_fast=False,
)
tokenizer.add_special_tokens(
    {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
    }
)


# ADJUST MODEL ARCHITECTURE HERE
model = AutoModelForCausalLM.from_pretrained(CONFIG["model_name_or_path"])
model_config = AutoConfig.from_pretrained(CONFIG["model_name_or_path"])

# model = AutoModelForCausalLM.from_config(model_config)
########################

model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

dataset = load_dataset(CONFIG["dataset_name"], CONFIG["dataset_name_config"])
dataset["train"] = dataset["train"].select(range(500_000))


remove_table = str.maketrans(
    "ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴáàảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ",
    "A" * 17
    + "D"
    + "E" * 11
    + "I" * 5
    + "O" * 17
    + "U" * 11
    + "Y" * 5
    + "a" * 17
    + "d"
    + "e" * 11
    + "i" * 5
    + "o" * 17
    + "u" * 11
    + "y" * 5,
)


def remove_accent(examples):
    if isinstance(examples, Sequence):
        return [remove_accent(e) for e in examples]
    elif isinstance(examples, dict):
        return {k: remove_accent(v) for k, v in examples.items()}
    elif isinstance(examples, str):
        return examples.translate(remove_table)
    else:
        return examples


def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples[CONFIG["text_column_name"]]])


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= CONFIG["block_size"]:
        total_length = (total_length // CONFIG["block_size"]) * CONFIG["block_size"]
    # Split by chunks of block_size.
    result = {
        k: [
            t[i : i + CONFIG["block_size"]]
            for i in range(0, total_length, CONFIG["block_size"])
        ]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


dataset = dataset.map(
    remove_accent,
    batched=True,
    num_proc=os.cpu_count() * 2,
)

dataset = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=dataset["train"].column_names,
).map(group_texts, batched=True, num_proc=4)

data = dataset["train"].train_test_split(test_size=0.1, seed=CONFIG["data_seed"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    data_collator=data_collator,
)

trainer.train()

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

trainer.push_to_hub()
