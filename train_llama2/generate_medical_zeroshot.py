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

def get_generations(prompts, temp, num_batches, batch_size):

    generations = []

    for i in tqdm(range(num_batches)):
        if i == num_batches - 1:
            texts = prompts[i*batch_size:]
        else:
            texts = prompts[i*batch_size: i*batch_size + batch_size]

        t1 = time.time()
        device = torch.device("cuda")
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)

        set_seed(i)
        outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=256, do_sample=True, top_p=0.9, num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id, remove_invalid_values=True, no_repeat_ngram_size=2, temperature=temp)
        print("Lenght of outputs is ", len(outputs))
        print("Time taken is ---- {}".format(time.time() - t1))

        for seq in outputs:
            text = tokenizer.decode(seq, skip_special_tokens=True)
            
            generations.append(text)
    
    return generations


def generate_zeroshot(pos_prompt, neg_prompt, neu_prompt, temps):

    num_generations = 60000
    pos_temp, neg_temp, neu_temp = temps[0], temps[1], temps[2]

    batch_size = 40
    num_batches = int(num_generations/batch_size)
    generations = []
    labels = []
    print("Number of batches are {}".format(num_batches))

    # first generate pos sentences
    label_prompts = [pos_prompt]*num_generations
    labels += ["positive"]*num_generations
    generations += get_generations(label_prompts, pos_temp, num_batches, batch_size)

    # generate neg sentences
    label_prompts = [neg_prompt]*num_generations
    labels += ["negative"]*num_generations
    generations += get_generations(label_prompts, neg_temp, num_batches, batch_size)

    # generate neu sentences
    label_prompts = [neu_prompt]*num_generations
    labels += ["neutral"]*num_generations
    generations += get_generations(label_prompts, neu_temp, num_batches, batch_size)

    print(len(generations), len(labels))

    return generations, labels



def generate_text(prompt):
    
    device = torch.device("cuda")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    for seed in range(10):
        t1 = time.time()

        set_seed(seed)

        outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=256, do_sample=True, top_p=0.9, num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id, remove_invalid_values=True, no_repeat_ngram_size=2, temperature=2.0)
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
    parser.add_argument("--input_path", type=str)

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

    # use temp 2.0 for this
    pos_prompt_medical = """<s>[INST] <<SYS>>
    You are a doctor who talks about medicine. Please only generate the sentence without any additional content before or after.
    <</SYS>>
    
    Please generate a single sentence indicating a positive sentiment with minimal noise.[/INST]"""

    # use temp 2.0 for this
    neg_prompt_medical = """<s>[INST] <<SYS>>
    You are a doctor who talks about medicine. Please only generate the sentence without any additional content before or after.
    <</SYS>>
    
    Please generate a single sentence indicating a negative sentiment with minimal noise.[/INST]"""

    # use temp 2.0 for this
    neu_prompt_medical = """<s>[INST] <<SYS>>
    You are a doctor who talks about medicine in a fact-based, and non-opinionated manner. Please don't involve emotional language or bias. Please only generate the sentence without any additional content before or after.
    <</SYS>>
    
    Please generate a single sentence with minimal noise. It should provide very general information.[/INST]"""

    # print(pos_prompt)
    # generate_text([pos_prompt_law])

    # exit()
    
    gens, labs = generate_zeroshot(pos_prompt_medical, neg_prompt_medical, neu_prompt_medical, [2.0, 2.0, 2.0])

    generations_dict = {"Texts": gens, "Labels": labs}
    df = pd.DataFrame(generations_dict)
    df.to_csv("{}/results/generations_zeroshot_medical.csv".format(repo_path), index=False)

    

if __name__ == "__main__":
    main()

# command: python train_llama2/generate_semeval_zeroshot.py --base_model meta-llama/Llama-2-13b-chat-hf