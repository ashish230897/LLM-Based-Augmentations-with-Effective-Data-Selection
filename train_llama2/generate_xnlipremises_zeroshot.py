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


def generate_domainaware_mnli_premises():

    total_generations = 60000
    
    domains = {"travel": int(0.4*total_generations), "government": int(0.4*total_generations), "fiction": int(0.2*total_generations)}
    
    labels = []
    for domain,count in domains.items():

        # prompt = "<s>[INST] <<SYS>>\nYour job is to generate a short sentence having less than thirty words\n<</SYS>>\n\nPlease generate a single sentence belonging to the {} domain. [/INST]".format(domain)
        if domain == "travel":
            prompt = "<s>[INST] <<SYS>>\nYou are a user who talks about other people's traveling experiences. Please only generate the traveling experience in a single sentence without any additional content before or after. \n<</SYS>>\n\nPlease generate a single and short sentence.[/INST]"
        else:
            prompt = "<s>[INST] <<SYS>>\nYour job is to generate a diverse sentence in the domain provided by the user. Please only generate the sentence without any additional content before or after. \n<</SYS>>\n\nPlease generate a single and short sentence belonging to the {} domain. [/INST]".format(domain)

        labels += [prompt]*(int(count))

    print("Length of labels is {}".format(len(labels)))
    batch_size = 80
    num_batches = int(total_generations/batch_size)
    generations = []

    for i in tqdm(range(num_batches)):
        if i == num_batches - 1:
            texts = labels[i*batch_size:]
        else:
            texts = labels[i*batch_size: i*batch_size + batch_size]

        t1 = time.time()
        device = torch.device("cuda")
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)

        set_seed(i)
        outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=128, do_sample=True, top_p=0.9, num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id, remove_invalid_values=True, no_repeat_ngram_size=2, temperature=1.7)
        print("Lenght of outputs is ", len(outputs))
        print("Time taken is ---- {}".format(time.time() - t1))

        for seq in outputs:
            text = tokenizer.decode(seq, skip_special_tokens=True)
            
            generations.append(text)

    generations_dict = {"Texts": generations}
    df = pd.DataFrame(generations_dict)
    df.to_csv("{}/results/generations_zeroshot_nlipremises.csv".format(repo_path), index=False)



def generate_text(prompt):
    
    device = torch.device("cuda")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    for seed in range(10):
        t1 = time.time()

        set_seed(seed)

        outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=128, do_sample=True, top_p=0.9, num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id, remove_invalid_values=True, no_repeat_ngram_size=2, temperature=1.7)
        print("Time taken for normal model is ---- {}".format(time.time() - t1))

        for seq in outputs:
            text = tokenizer.decode(seq, skip_special_tokens=True)
            print(text)
            #text = text.split(":")[1]
            #print(text)
            


def main():

    global tokenizer
    global inference_model
    global model

    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_path", type=str, required=False)
    parser.add_argument("--load_finetuned", action="store_true")
    parser.add_argument("--base_model", type=str)

    args = parser.parse_args()
    # base_model = "meta-llama/Llama-2-7b-chat-hf"
    base_model = args.base_model    

    saved_path = args.saved_path
    if saved_path is not None:
        config = PeftConfig.from_pretrained(saved_path)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            trust_remote_code=True,
            use_auth_token="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",
            device_map="auto"
        )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_auth_token="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")
    tokenizer.pad_token = tokenizer.eos_token

    # Load the LoRA model
    if args.load_finetuned:
        inference_model = PeftModel.from_pretrained(model, saved_path)

    #prompt = "<s>[INST] <<SYS>>\nYour job is to generate a diverse sentence in the domain provided by the user. Please only generate the sentence without any additional content before or after. \n<</SYS>>\n\nPlease generate a single and short sentence belonging to the {} domain.[/INST]".format("travel")
    prompt = "<s>[INST] <<SYS>>\nYou are a user who talks about other people's traveling experiences. Please only generate the traveling experience in a single sentence without any additional content before or after. \n<</SYS>>\n\nPlease generate a single and short sentence.[/INST]".format("travel")
    #generate_text([prompt])

    #exit()

    generate_domainaware_mnli_premises()

    

if __name__ == "__main__":
    main()

#python train_llama2/generate_xnlipremises_zeroshot.py --base_model meta-llama/Llama-2-13b-chat-hf