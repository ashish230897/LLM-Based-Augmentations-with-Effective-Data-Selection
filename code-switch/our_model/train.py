from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset,DatasetDict
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_int8_training
import random
import numpy as np
import os
from transformers import DataCollatorForSeq2Seq, HfArgumentParser, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer
from dataclasses import dataclass, field
from typing import Optional
import wandb

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
# os.environ["WANDB_DISABLED"] = "true"

repo_path = os.getcwd()

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)
model_args, data_args, training_args = None, None, None
tokenizer, model = None, None

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@dataclass
class ModelArguments:
    # Base model parameters
    model_name_or_path: Optional[str] = field(default="bigscience/mt0-large")
    
    load_in_8bit: bool = field(
                default=False, metadata={"help": "Whether to convert the loaded model into mixed-8bit quantized model."}
    )

    device_map: Optional[str] = field(default="auto")
    
    # LoRA parameters
    lora_r: int = field(default=64, metadata={"help": "Lora rank."})
    lora_alpha: int = field(default=32, metadata={"help": "Lora alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "Lora dropout."})

    wandb_project: Optional[str] = field(default="sft-model-training")


@dataclass
class DataArguments:
    
    train_file_path: Optional[str] = field(default='./data/train.csv', metadata={"help": "Path to the training file."})
    valid_file_path: Optional[str] = field(default='./data/valid.csv', metadata={"help": "Path to the training file."})
    max_seq_length_input: Optional[int] = field(
        default=256, metadata={"help": "Maximum length of source. Sequences will be right padded (and possibly truncated)."}
    )
    max_seq_length_output: Optional[int] = field(
        default=256, metadata={"help": "Maximum length of target. Sequences will be right padded (and possibly truncated)."}
    )    


def preprocess_function(sample, padding="max_length"):
    
    inputs = [item for item in sample["inputs"]]

    # not padding to max length for now

    # tokenize inputs
    # model_inputs = tokenizer(inputs, max_length=data_args.max_seq_length_input, padding=padding, truncation=True)
    model_inputs = tokenizer(inputs, max_length=data_args.max_seq_length_input, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    # labels = tokenizer(text_target=sample["outputs"], max_length=data_args.max_seq_length_output, padding=padding, truncation=True)
    labels = tokenizer(text_target=sample["outputs"], max_length=data_args.max_seq_length_output, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    # if padding == "max_length":
    #     labels["input_ids"] = [
    #         [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    #     ]

    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs


def train_model(train_df, valid_df):

    global tokenizer
    global model

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=model_args.load_in_8bit,
        trust_remote_code=True,
        use_auth_token="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",
        device_map=model_args.device_map
    )
    model.config.use_cache = False

    if model_args.load_in_8bit:
        model = prepare_model_for_int8_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, 
                                            use_auth_token="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")
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        r=model_args.lora_r,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=["q", "k", "v"]
    )

    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()
    print(model)

    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)

    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["inputs", "outputs"])
    tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True, remove_columns=["inputs", "outputs"])

    label_pad_token_id = -100
    
    # Data collator
    # this will pad to the longest sequence in the batch (by default padding is longest in the batch)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )

    print("Number of train examples are {}".format(len(train_df)))
    
    trainer = Seq2SeqTrainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        #compute_metrics=lambda x: mt.compute_metrics(x, tokenizer, data_args) if training_args.predict_with_generate else None,
        #compute_metrics=mt.compute_metrics
    )

    if model_args.load_in_8bit:
        for name, module in trainer.model.named_modules():
            if "norm" in name:
                module = module.to(torch.float32)

    print("Length of train dataset is ", len(train_df))
    print("Length of valid dataset is ", len(valid_df))
    
    # start training
    trainer.train()

    # saving best model at the end
    trainer.save_model(os.path.join(training_args.output_dir,'final_model'))


def main():

    global data_args
    global training_args
    global model_args

    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(42)
    
    # create output dir if it does not exists
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    train_df = pd.read_csv(data_args.train_file_path)
    valid_df = pd.read_csv(data_args.valid_file_path)

    print(model_args)

    print()

    print(data_args)

    print()

    print(training_args)

    run = wandb.init(
        # Set the project where this run will be logged
        project=model_args.wandb_project,
        # Track hyperparameters and run metadata
        config={
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "grad_acc_steps": training_args.gradient_accumulation_steps
        })

    train_model(train_df, valid_df)


if __name__ == "__main__":
    main()
