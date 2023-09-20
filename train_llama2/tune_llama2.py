from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset,DatasetDict
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, HfArgumentParser
from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer
import random
import numpy as np
import os
import time
from dataclasses import dataclass, field
from typing import Optional


os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
os.environ["WANDB_DISABLED"] = "true"

repo_path = os.getcwd()

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)

model_args, data_args, training_args = None, None, None

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@dataclass
class ModelArguments:
    # Base model parameters
    model_name_or_path: Optional[str] = field(default="bigscience/mt0-large")
    
    load_in_4bit: bool = field(
                default=False, metadata={"help": "Whether to convert the loaded model into mixed-8bit quantized model."}
    )

    device_map: Optional[str] = field(default="auto")
    
    # LoRA parameters
    bnb_4bit_quant_type: str = field(default="nf4")
    lora_r: int = field(default=64, metadata={"help": "Lora rank."})
    lora_alpha: int = field(default=32, metadata={"help": "Lora alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "Lora dropout."})


@dataclass
class DataArguments:
    
    file_path: Optional[str] = field(default='data/en/wiki_auto/train_mt0.csv', metadata={"help": "Path to the training file."})
    max_seq_length: Optional[int] = field(
        default=256)


def train_model(train_df, test_df):

    bnb_config = None
    if model_args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    print("quantization config is ", bnb_config)

    model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            use_auth_token="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",
            device_map=model_args.device_map
        )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, use_auth_token="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")
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        r=model_args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj"]
    )

    print("output dir is {}".format(training_args.output_dir))
    print("Number of train examples are {}".format(len(train_df)))

    dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(test_df)
    })

    print("Pad token is {}".format(tokenizer.pad_token))
    print("EOS token is {}".format(tokenizer.pad_token))
    print()

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['test'],
        peft_config=peft_config,
        dataset_text_field="text_label",
        max_seq_length=data_args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    if model_args.load_in_4bit:
        for name, module in trainer.model.named_modules():
            if "norm" in name:
                module = module.to(torch.float32)

    
    # start training
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir,'final_model'))


def main():

    global data_args
    global model_args
    global training_args

    set_seed(42)

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print("Arguments are: ")
    print(model_args)
    print()
    print(data_args)
    print()
    print(training_args)
    print()


    # create output dir if it does not exists
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    df = pd.read_csv(data_args.file_path)

    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

    train_model(train_df, test_df)


if __name__ == "__main__":
    main()

