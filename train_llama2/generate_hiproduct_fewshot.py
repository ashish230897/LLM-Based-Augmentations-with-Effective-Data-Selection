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

    num_generations = 250000

    label_prompts = [pos_prompt, neg_prompt, neu_prompt, neu_prompt]*int(num_generations/4)
    labels = ["positive", "negative", "neutral", "neutral"]*int(num_generations/4)
    
    print(len(labels))

    print("Length of labels is {}".format(len(labels)))
    batch_size = 50
    num_batches = int(len(labels)/batch_size)
    generations = []

    for i in tqdm(range(num_batches)):
        if i == num_batches - 1:
            texts = label_prompts[i*batch_size:]
        else:
            texts = label_prompts[i*batch_size: i*batch_size + batch_size]

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

    generations_dict = {"Texts": generations, "Labels": labels}
    df = pd.DataFrame(generations_dict)
    df.to_csv("{}/results/generations_fewshot_hiproductv2.csv".format(repo_path), index=False)



def generate_text(prompt):
    
    device = torch.device("cuda")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    for seed in range(5):
        t1 = time.time()

        set_seed(seed)

        outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=400, do_sample=True, top_p=0.9, num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id, remove_invalid_values=True, no_repeat_ngram_size=2, temperature=1.5)
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
    You are a user providing reviews on travels, movies and various electronic gadgets. Please only generate the review without any additional content before or after.

    <</SYS>>

    Please generate a single review in not more than two short sentences on one of the system specified products/movies/travels indicating a positive sentiment, and similar in style to the below five few-shot reviews (Do not ignore the sentences in the last). Here are the five few-shot reviews:
    Review: The sanctuary is an ideal place for those interested in birds.
    Review: The actor, who appears in a small role as Tiwari, is good at his goof-ups.
    Review: These are the things that make the film interesting as well as fun.
    Review: The Kashmiri Pashmina shawls, stoles, and wood crafts of Srinagar are famous throughout the world.
    Review: It has a good display and a good viewing angle.
    Review: It has a 500 GB hard disk and two GB of RAM which optimizes the working speed with a 2. 30 GHz Intel Pentium processor.
    Review: Its display is better than all its rivals in this price range.
    Review: [/INST]
    """

    neg_prompt = """<s>[INST] <<SYS>>
    You are a user providing reviews on travels, movies and various electronic gadgets. Please only generate the review without any additional content before or after.

    <</SYS>>

    Please generate a single review in not more than two short sentences on one of the system specified products/movies/travels indicating a negative sentiment, and similar in style to the below five few-shot reviews (Do not ignore the sentences in the last). Here are the five few-shot reviews:
    Review: Due to which the beauty of the Taj is being destroyed.
    Review: The pace of the film is quite slow before the interval.
    Review: Deepak Dobriyal is seen overacting a couple of scenes.
    Review: The parliamentary committee that came for inspection on Saturday had also expressed concern over the destruction of the likeness of the Taj when its mosaic was destroyed.
    Review: One drawback of the game is that it comes in only one graphic mode.
    Review: At first glance or in an advertisement, this phone will win your heart, but after a while of use, small flaws in its design will start to appear.
    Review: It seems to be running a bit slow due to the weak processor.
    Review: [/INST]
    """

    # neu_prompt = """<s>[INST] <<SYS>>
    # You are a user providing reviews on travels, movies and various electronic gadgets. Please only generate the review without any additional content before or after.

    # <</SYS>>

    # Please generate a single review in not more than two short sentences on one of the system specified products/movies/travels indicating neither negative nor positive sentiment, and similar in style to the below five few-shot reviews (Do not ignore the sentences in the last). Here are the five few-shot reviews:
    # Review: The Sundarbans National Park is located in the Sundarbans delta region of the Ganges River in the southern part of West Bengal state.
    # Review: In the process of making the film and living his character, many times the actors are influenced and disturbed by the essence of the film, but only a few of them are able to convey the zest of the role.
    # Review: The character of the villain is also very similar to the old films.
    # Review: There are many institutions in Bodh Gaya that do the same.
    # Review: If you have an Android-based handset, it can be purchased as a second handset.
    # Review: The closest competitor to the Asus Fonepad tablet is the Samsung Galaxy Tab 2.
    # Review: The company is offering a 12-month warranty on the device.
    # Review: [/INST]
    # """

    neu_prompt = """<s>[INST] <<SYS>>
    You are a user who talks about travels, movies and various electronic gadgets in a fact-based, and non-opinionated manner. Please don't involve emotional language or bias. Please only generate the description without any additional content before or after.

    <</SYS>>

    Please generate a single and very short sentence on one of the system specified products/movies/travels. It should provide very general information, and should be similar in style to the below five few-shot sentences (Do not ignore the sentences in the last). Here are the five few-shot sentences:
    Sentence: The Sundarbans National Park is located in the Sundarbans delta region of the Ganges River in the southern part of West Bengal state.
    Sentence: In the process of making the film and living his character, many times the actors are influenced and disturbed by the essence of the film, but only a few of them are able to convey the zest of the role.
    Sentence: There are many institutions in Bodh Gaya that do the same.
    Sentence: If you have an Android-based handset, it can be purchased as a second handset.
    Sentence: The closest competitor to the Asus Fonepad tablet is the Samsung Galaxy Tab 2.
    Sentence: [/INST]
    """

    # print(prompt)
    # generate_text([neu_prompt])

    # exit()

    generate_fewshot(pos_prompt, neg_prompt, neu_prompt)

    

if __name__ == "__main__":
    main()