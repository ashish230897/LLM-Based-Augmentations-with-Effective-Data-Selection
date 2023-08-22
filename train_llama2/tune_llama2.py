from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset,DatasetDict
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import transformers
from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer
import random
import numpy as np
import os
import time
import argparse


os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
os.environ["WANDB_DISABLED"] = "true"

repo_path = "/raid/speech/ashish/TSTG_new"

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

model_name = "meta-llama/Llama-2-7b-hf"
    
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        use_auth_token="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",
        device_map="auto"
    )
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_auth_token="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")
tokenizer.pad_token = tokenizer.eos_token


# taking lora config from huggingface repo: https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing#scrollTo=Ybeyl20n3dYH
lora_alpha = 32
lora_dropout = 0.05
lora_r = 8

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj"]
)

output_dir = None
per_device_train_batch_size = 16
gradient_accumulation_steps = 4  # making the effective batch size as 64
optim = "paged_adamw_8bit"  # following hf repo hyper params
save_steps = 20
eval_steps = 20
logging_steps = 1
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 120
warmup_ratio = 0.03
lr_scheduler_type = "constant"
num_train_epochs = 10

max_seq_length = 512

def generate_dummy():

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )

    sequences = pipeline(
        ["<s>"],
        max_length=100,
        do_sample=True,
        top_k=20,
        top_p=0.95,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=True
    )
    for seq in sequences:
        print(f"Result: {seq[0]['generated_text']}")


def train_model(train_df, test_df):

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=True,
        # max_steps=max_steps,
        max_grad_norm=max_grad_norm,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        metric_for_best_model="eval_loss",
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
    )

    print("output dir is {}".format(output_dir))
    print("Number of train examples are {}".format(len(train_df)))

    dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(test_df)
    })

    print("Pad token is {}".format(tokenizer.pad_token))
    print()
    print("Dummy generation before fine-tuning----------")
    t1 = time.time()
    generate_dummy()
    print("----------------- Time taken is {}".format(time.time()-t1))
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['test'],
        peft_config=peft_config,
        dataset_text_field="text_label",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    
    # start training
    trainer.train()
    trainer.save_model(os.path.join(output_dir,'final_model'))


def main():

    global output_dir

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--seed",
        type=int,
        required=True
    )

    parser.add_argument(
        "--size",
        type=int,
        required=True
    )

    args = parser.parse_args()

    set_seed(42)

    output_dir = "{}/results/llama2/xnli_seed{}_size{}_label".format(repo_path, args.seed, args.size)
    
    # create output dir if it does not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv("{}/data/xnli/train-ensubsetseed{}size{}_texts.csv".format(repo_path, args.seed, args.size))

    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

    train_model(train_df, test_df)


if __name__ == "__main__":
    main()

