from sklearn.model_selection import train_test_split
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, set_seed
import transformers
from peft import PeftConfig, PeftModel, get_peft_model, LoraConfig, PeftModelForCausalLM
import pandas as pd
from tqdm import tqdm
import time
import argparse
import os

repo_path = os.getcwd()

tokenizer = None
inference_model = None
model = None

def generate_pipeline():

    pipeline = transformers.pipeline(
        "text-generation",
        model=inference_model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        device_map="auto",
    )

    premises, hypos = [], []
    # num_required_generations = 2

    batch_size = 10
    num_batches = 3
    print("Number of batches are {}".format(num_batches))

    for i in tqdm(range(num_batches)):
        print("Processing batch {}".format(i))
        set_seed(i)
        curr_batch = ["<s>"]*batch_size
        
        t1 = time.time()
        sequences = pipeline(
            curr_batch,
            max_length=50,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )
        print("Time taken for generation is {}".format(time.time() - t1))

        t1 = time.time()
        for seq in sequences:
            text = seq[0]['generated_text']

            print(text)
            
            lines = text.split("\n")
            pre, hyp = False, False
            premise, hypothesis = "", ""
            for line in lines:
                if "Premise:" in line: 
                    premise = " ".join(line.split(":")[1:]).strip().replace("\n", "")
                    pre = True
                if "Hypothesis:" in line: 
                    hypothesis = " ".join(line.split(":")[1:]).strip().replace("\n", "")
                    hyp = True
                
                if pre and hyp: break

            premises.append(premise)
            hypos.append(hypothesis)
        
        print("Time taken for collating is {}".format(time.time() - t1))

    generations_dict = {"Premises": premises, "Hypothesis": hypos}
    df = pd.DataFrame(generations_dict)
    df.to_csv("{}/results/generations.csv".format(repo_path), index=False)

def generate_generate():

    num_batches = 3
    batch_size = 100
    premises, hypos = [], []

    for i in tqdm(range(num_batches)):
        texts = ["<s>"]*batch_size

        t1 = time.time()
        device = torch.device("cuda")
        inputs = tokenizer(texts, return_tensors="pt").to(device)
        
        set_seed(i)
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, top_p=0.5, early_stopping=True, num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id, remove_invalid_values=True)
        print("Lenght of outputs is ", len(outputs))
        print("Time taken is ---- {}".format(time.time() - t1))

        for seq in outputs:
            text = tokenizer.decode(seq, skip_special_tokens=True)

            lines = text.split("\n")
            pre, hyp = False, False
            premise, hypothesis = "", ""
            for line in lines:
                if "Premise:" in line: 
                    premise = " ".join(line.split(":")[1:]).strip().replace("\n", "")
                    pre = True
                if "Hypothesis:" in line: 
                    hypothesis = " ".join(line.split(":")[1:]).strip().replace("\n", "")
                    hyp = True
                
                if pre and hyp: break

            premises.append(premise)
            hypos.append(hypothesis)

    generations_dict = {"Premises": premises, "Hypothesis": hypos}
    df = pd.DataFrame(generations_dict)
    df.to_csv("{}/results/generations.csv".format(repo_path), index=False)


def generate_labelaware(seed, size):

    total_generations = 150000
    labels = ["entailment \n", "contradiction \n", "neutral \n"]*(int(total_generations/3))
    print("Length of labels is {}".format(len(labels)))
    batch_size = 100
    num_batches = int(total_generations/batch_size)
    premises, hypos = [], []

    for i in tqdm(range(num_batches)):
        texts = [labels[i]]*batch_size

        t1 = time.time()
        device = torch.device("cuda")
        inputs = tokenizer(texts, return_tensors="pt").to(device)

        set_seed(i)
        outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=100, do_sample=True, top_p=0.5, early_stopping=True, num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id, remove_invalid_values=True)
        print("Lenght of outputs is ", len(outputs))
        print("Time taken is ---- {}".format(time.time() - t1))

        for seq in outputs:
            text = tokenizer.decode(seq, skip_special_tokens=True)

            lines = text.split("\n")
            pre, hyp = False, False
            premise, hypothesis = "", ""
            for line in lines:
                if "Premise:" in line: 
                    premise = " ".join(line.split(":")[1:]).strip().replace("\n", "")
                    pre = True
                if "Hypothesis:" in line: 
                    hypothesis = " ".join(line.split(":")[1:]).strip().replace("\n", "")
                    hyp = True
                
                if pre and hyp: break

            premises.append(premise)
            hypos.append(hypothesis)

    generations_dict = {"Premises": premises, "Hypothesis": hypos}
    df = pd.DataFrame(generations_dict)
    df.to_csv("{}/results/generations_label_seed{}size{}.csv".format(repo_path, seed, size), index=False)

def generate_text(prompt):
    
    device = torch.device("cuda")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    set_seed(42)
    t1 = time.time()
    outputs = inference_model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=100, do_sample=True, top_p=0.5, early_stopping=True, num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id, remove_invalid_values=True)
    print("Time taken for peft model is ---- {}".format(time.time() - t1))

    for seq in outputs:
        text = tokenizer.decode(seq, skip_special_tokens=True)
        print(text)
    
    t1 = time.time()
    outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=100, do_sample=True, top_p=0.5, early_stopping=True, num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id, remove_invalid_values=True)
    print("Time taken for normal model is ---- {}".format(time.time() - t1))

    for seq in outputs:
        text = tokenizer.decode(seq, skip_special_tokens=True)
        print(text)


def main():

    global tokenizer
    global inference_model
    global model

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--size", type=int, required=True)

    args = parser.parse_args()

    saved_path = "{}/results/llama2/xnli_seed{}_size{}_label/final_model/".format(repo_path, args.seed, args.size)

    print("saved path is ", saved_path)

    config = PeftConfig.from_pretrained(saved_path)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print("base model name is ", config.base_model_name_or_path)
    
    model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            use_auth_token="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",
            device_map="auto"
        )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True, use_auth_token="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")
    tokenizer.pad_token = tokenizer.eos_token

    # print("Model before peft loading")
    # print(model)
    # print()

    print(config.task_type)

    # Load the LoRA model
    inference_model = PeftModel.from_pretrained(model, saved_path)
    # print("Inference model is: ")
    # print(inference_model)
    # print()

    # print("Model after inplace loading is ")
    # print(model)
    # print()

    # generate_pipeline()
    # generate_generate()

    #generate_text(["entailment \n"])
    # generate_text(["contradiction \n"])
    # generate_text(["neutral \n"])

    generate_labelaware(args.seed, args.size)

    

if __name__ == "__main__":
    main()