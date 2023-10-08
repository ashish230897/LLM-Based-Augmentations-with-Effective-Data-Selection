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


def generate_fewshot(pos_prompt, neg_prompt, neu_prompt):

    num_generations = 200000

    labels = [pos_prompt, neg_prompt, neu_prompt]*int(num_generations/3)
    
    print(len(labels))

    print("Length of labels is {}".format(len(labels)))
    batch_size = 70
    num_batches = int(len(labels)/batch_size)
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
        outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=400, do_sample=True, top_p=0.9, num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id, remove_invalid_values=True, no_repeat_ngram_size=2, temperature=1.5)
        print("Lenght of outputs is ", len(outputs))
        print("Time taken is ---- {}".format(time.time() - t1))

        for seq in outputs:
            text = tokenizer.decode(seq, skip_special_tokens=True)
            
            generations.append(text)

    generations_dict = {"Texts": generations}
    df = pd.DataFrame(generations_dict)
    df.to_csv("{}/results/generations_fewshot_bnsentiment.csv".format(repo_path), index=False)



def generate_text(prompt):
    
    device = torch.device("cuda")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    for seed in range(10):
        t1 = time.time()

        set_seed(seed)

        outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=400, do_sample=True, top_p=0.9, num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id, remove_invalid_values=True, no_repeat_ngram_size=2, temperature=1.5)#temperature=1.3)
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

    pos_prompt = """<s>[INST] <<SYS>>
    You are a user providing reviews on news across a range of subjects, including politics, sports, food, entertainment, education, lifestyle, travel, fashion, agriculture etc. Please only generate the review without any additional content before or after.
    <</SYS>>

    Please generate a single review in not more than two short sentences on news in one of the system specified subjects, indicating a positive sentiment, and similar in style to the below five few-shot reviews. Here are the five few-shot reviews:
    Review: I want to see Sourav Ganguly as the head coach of the Bangladesh team, if everyone agrees, like it.
    Review: There will be something to talk about, Mamu. Very nice food.
    Review: I don't just want to thank someone, I want to say that God helps those who help others.
    Review: Bhaiya is more of a foodie with a little loose dress, along with your food jokes, but your physical expressions also make us crave for food, hope you like it, thank you.
    Review: Many heroes have come out of the slums in the world.   
    Review: [/INST]
    """

    neg_prompt = """<s>[INST] <<SYS>>
    You are a user providing reviews on news across a range of subjects, including politics, sports, food, entertainment, education, lifestyle, travel, fashion, agriculture etc. Please only generate the review without any additional content before or after.
    <</SYS>>

    Please generate a single review in not more than two short sentences on news in one of the system specified subjects, indicating a negative sentiment, and similar in style to the below five few-shot reviews. Here are the five few-shot reviews:
    Review: Don't agree Shraddha Kapoor. I want to see another heroine.
    Review: Feeling very tired due to lack of hair.
    Review: Everyone wants to take advantage of people's helplessness.
    Review: Where to go in Bangladesh Living in a country where you won't get justice.
    Review: Bengalis do not value their mother tongue.
    Review: [/INST]
    """

    neu_prompt = """<s>[INST] <<SYS>>
    You are a user providing reviews on news across a range of subjects, including politics, sports, food, entertainment, education, lifestyle, travel, fashion, agriculture etc. Please only generate the review without any additional content before or after.
    <</SYS>>

    Please generate a single review in not more than two short sentences on news in one of the system specified subjects, indicating neither negative nor positive sentiment, and similar in style to the below five few-shot reviews. Here are the five few-shot reviews:
    Review: Along with the students, the youth, the farmers, the wage-earners, the labor teachers, everyone will think about the country.
    Review: I'm not really clear. What do you really mean? What did I say in the video?
    Review: Remove the elephant from the fire if you want to save the country.
    Review: You will find Biryani House in Lalbagh.
    Review: He is a former professional tennis player of Rajshahi Division and currently a dealer of Teletalk and DBBL in Rajshahi District.
    Review: [/INST]
    """

    # print(prompt)
    # generate_text([neg_prompt])

    # exit()

    generate_fewshot(pos_prompt, neg_prompt, neu_prompt)

    

if __name__ == "__main__":
    main()