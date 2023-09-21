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
from transformers import LogitsProcessor,LogitsProcessorList, ForcedEOSTokenLogitsProcessor

repo_path = os.getcwd()

tokenizer = None
inference_model = None
model = None


class EosTokenRewardLogitsProcessor(LogitsProcessor):
  
  def __init__(self,  eos_token_id: int, max_length: int):
    
        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(f"`eos_token_id` has to be a positive integer, but is {eos_token_id}")

        if not isinstance(max_length, int) or max_length < 1:
          raise ValueError(f"`max_length` has to be a integer bigger than 1, but is {max_length}")

        self.eos_token_id = eos_token_id
        self.max_length=max_length

  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    
    cur_len = input_ids.shape[-1]
    
    # start to increase the reward of the  eos_tokekn from 80% max length  progressively on length
    for cur_len in (max(0,int(self.max_length*0.8)), self.max_length ):
      ratio = cur_len/self.max_length
      num_tokens = scores.shape[1] # size of vocab
      scores[:, [i for i in range(num_tokens) if i != self.eos_token_id]] =\
      scores[:, [i for i in range(num_tokens) if i != self.eos_token_id]]*ratio*10*torch.exp(-torch.sign(scores[:, [i for i in range(num_tokens) if i != self.eos_token_id]]))
      scores[:, self.eos_token_id] = 120*ratio
    
    return scores


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

def generate_labelaware_bnsentiment():

    total_generations = 150000
    # total_generations = 15
    labels = ["positive \n", "negative \n", "neutral \n"]*(int(total_generations/3))
    print("Length of labels is {}".format(len(labels)))
    batch_size = 100
    num_batches = int(total_generations/batch_size)
    generations = []

    for i in tqdm(range(num_batches)):
        texts = [labels[i]]*batch_size

        t1 = time.time()
        device = torch.device("cuda")
        inputs = tokenizer(texts, return_tensors="pt").to(device)

        set_seed(i)
        outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=44, do_sample=True, top_p=0.8, num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id, remove_invalid_values=True, no_repeat_ngram_size=2)
        print("Lenght of outputs is ", len(outputs))
        print("Time taken is ---- {}".format(time.time() - t1))

        for seq in outputs:
            text = tokenizer.decode(seq, skip_special_tokens=True)

            lines = text.split("\n")
            text = ".".join(lines[1].split(".")[:-1]) + "."
            generations.append(text)

    generations_dict = {"Texts": generations}
    df = pd.DataFrame(generations_dict)
    df.to_csv("{}/results/generations_label_bnsentimentv3.csv".format(repo_path), index=False)


def generate_domainaware_bnsentiment():

    total_generations = 200000
    
    domains = {"politics": int(0.2*total_generations), "sports": int(0.1*total_generations), "food": int(0.1*total_generations), 
               "entertainment": int(0.1*total_generations), 
               "lifestyle": int(0.1*total_generations), "education": int(0.1*total_generations), 
               "travelling": int(0.1*total_generations), "fashion": int(0.1*total_generations), 
               "agriculture": int(0.1*total_generations)}
    
    labels = []
    for domain,count in domains.items():

        # "<s>[INST] <<SYS>>\nYou have to generate text that belongs to {} domain\n<</SYS>>\n\nGenerate a positive sentiment sentence [/INST]".format(domain)
        
        # pos_prompt = "<s>[INST] <<SYS>>\nYou are a user commenting on the {} news, just give the comment without telling anything else\n<</SYS>>\n\nPositive sentiment sentence is: [/INST]".format(domain)
        # neg_prompt = "<s>[INST] <<SYS>>\nYou are a user commenting on the {} news, just give the comment without telling anything else\n<</SYS>>\n\nNegative sentiment sentence is: [/INST]".format(domain)
        # neu_prompt = "<s>[INST] <<SYS>>\nYou are a user commenting on the {} news, just give the comment without telling anything else\n<</SYS>>\n\nNeutral sentiment sentence is: [/INST]".format(domain)
        # pos_prompt = "<s>[INST] <<SYS>>\nYou are a user commenting on the {} news, just give the answer without telling anything else\n<</SYS>>\n\nPlease generate a sentence indicating a positive sentiment.\nSentence: [/INST]".format(domain)
        # neg_prompt = "<s>[INST] <<SYS>>\nYou are a user commenting on the {} news, just give the answer without telling anything else\n<</SYS>>\n\nPlease generate a sentence indicating a negative sentiment.\nSentence: [/INST]".format(domain)
        # neu_prompt = "<s>[INST] <<SYS>>\nYou are a user commenting on the {} news, just give the answer without telling anything else\n<</SYS>>\n\nPlease generate a sentence indicating a neutral sentiment.\nSentence: [/INST]".format(domain) 
        # pos_prompt = '<s>[INST] <<SYS>>\nYou are a user who likes commenting on the {} news\n<</SYS>>\n\nPlease generate a single sentence indicating a positive sentiment as in the example below:\nSentence: "Today the weather is very good."\nSentence: [/INST]'.format(domain)
        # neg_prompt = '<s>[INST] <<SYS>>\nYou are a user who likes commenting on the {} news\n<</SYS>>\n\nPlease generate a single sentence indicating a negative sentiment as in the example below:\nSentence: "Today the weather is very harsh."\nSentence: [/INST]'.format(domain)
        # neu_prompt = '<s>[INST] <<SYS>>\nYou are a user who likes commenting on the {} news\n<</SYS>>\n\nPlease generate a single sentence indicating a neutral sentiment as in the example below:\nSentence: "Today the weather is slightly sunny."\nSentence: [/INST]'.format(domain)
        pos_prompt = "<s>[INST] <<SYS>>\nYour job is to comment on news from {}.\n<</SYS>>\n\nPlease share a review in one sentence expressing a positive sentiment. Please only share the review sentence without any additional content before or after. Please make sure the review is a meaningful sentence.[/INST]".format(domain)
        neg_prompt = "<s>[INST] <<SYS>>\nYour job is to comment on news from {}.\n<</SYS>>\n\nPlease share a review in one sentence expressing a negative sentiment. Please only share the review sentence without any additional content before or after. Please make sure the review is a meaningful sentence.[/INST]".format(domain)
        neu_prompt = "<s>[INST] <<SYS>>\nYour job is to comment on news from {}.\n<</SYS>>\n\nPlease share a review in one sentence expressing a neutral sentiment. Please only share the review sentence without any additional content before or after. Please make sure the review is a meaningful sentence.[/INST]".format(domain)
        # pos_prompt = "<s>[INST] <<SYS>>\nYour job is to comment on news from {}.\n<</SYS>>\n\nPlease share a meaningful one-sentence review conveying a positive sentiment, and please provide only the review sentence without any preceding or following content.[/INST]".format(domain)
        # neg_prompt = "<s>[INST] <<SYS>>\nYour job is to comment on news from {}.\n<</SYS>>\n\nPlease share a meaningful one-sentence review conveying a negative sentiment, and please provide only the review sentence without any preceding or following content.[/INST]".format(domain)
        # neu_prompt = "<s>[INST] <<SYS>>\nYour job is to comment on news from {}.\n<</SYS>>\n\nPlease share a meaningful one-sentence review conveying a neutral sentiment, and please provide only the review sentence without any preceding or following content.[/INST]".format(domain)

        labels += [pos_prompt, neg_prompt, neu_prompt]*(int(count/3))
    
    #print(labels)

    print("Length of labels is {}".format(len(labels)))
    batch_size = 200
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
            eos_token_id=tokenizer.eos_token_id, remove_invalid_values=True, no_repeat_ngram_size=2, temperature=1.5)
        print("Lenght of outputs is ", len(outputs))
        print("Time taken is ---- {}".format(time.time() - t1))

        for seq in outputs:
            text = tokenizer.decode(seq, skip_special_tokens=True)

            #if len(text.split('"')) >= 4:
            # if '"' in text:
            #     # text = text.split('"')[3]
            #     text = text.split('"')[1]
            # # elif ":" in text:
            # #     text = text.split(':')[1]
            # else:
            #     text = text.split('[/INST]')[1]
            
            generations.append(text)

    generations_dict = {"Texts": generations}
    df = pd.DataFrame(generations_dict)
    df.to_csv("{}/results/generations_label_sst5chat_7bquantized.csv".format(repo_path), index=False)


def generate_labelaware_mlheadline():

    total_generations = 150000
    # total_generations = 15
    labels = ["entertainment \n", "sports \n", "business \n"]*(int(total_generations/3))
    print("Length of labels is {}".format(len(labels)))
    batch_size = 100
    num_batches = int(total_generations/batch_size)
    generations = []

    for i in tqdm(range(num_batches)):
        texts = [labels[i]]*batch_size

        t1 = time.time()
        device = torch.device("cuda")
        inputs = tokenizer(texts, return_tensors="pt").to(device)

        set_seed(i)
        outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=30, do_sample=True, top_p=0.5, early_stopping=True, num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id, remove_invalid_values=True, no_repeat_ngram_size=2)
        # outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=30, do_sample=True, top_k=50, early_stopping=True, num_return_sequences=1,
        #     eos_token_id=tokenizer.eos_token_id, remove_invalid_values=True, no_repeat_ngram_size=2)
        print("Lenght of outputs is ", len(outputs))
        print("Time taken is ---- {}".format(time.time() - t1))

        for seq in outputs:
            text = tokenizer.decode(seq, skip_special_tokens=True)

            lines = text.split("\n")
            text = ". ".join(lines[1].split(".")[:-1]) + "."

            generations.append(text)

    generations_dict = {"Texts": generations}
    df = pd.DataFrame(generations_dict)
    df.to_csv("{}/results/generations_label_mlheadline.csv".format(repo_path), index=False)

def generate_labelaware_hiproduct():

    total_generations = 150000
    # total_generations = 15
    labels = ["positive \n", "negative \n", "neutral \n"]*(int(total_generations/3))
    print("Length of labels is {}".format(len(labels)))
    batch_size = 100
    num_batches = int(total_generations/batch_size)
    generations = []

    for i in tqdm(range(num_batches)):
        texts = [labels[i]]*batch_size

        t1 = time.time()
        device = torch.device("cuda")
        inputs = tokenizer(texts, return_tensors="pt").to(device)

        set_seed(i)
        outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=30, do_sample=True, top_p=0.5, early_stopping=True, num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id, remove_invalid_values=True, no_repeat_ngram_size=2)
        # outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=30, do_sample=True, top_k=50, early_stopping=True, num_return_sequences=1,
        #     eos_token_id=tokenizer.eos_token_id, remove_invalid_values=True, no_repeat_ngram_size=2)
        print("Lenght of outputs is ", len(outputs))
        print("Time taken is ---- {}".format(time.time() - t1))

        for seq in outputs:
            text = tokenizer.decode(seq, skip_special_tokens=True)

            #print(text)

            lines = text.split("\n")
            text = ". ".join(lines[1].split(".")[:-1]) + "."

            #print(text)

            generations.append(text)

    generations_dict = {"Texts": generations}
    df = pd.DataFrame(generations_dict)
    df.to_csv("{}/results/generations_label_hiproduct.csv".format(repo_path), index=False)

def generate_text(prompt):
    
    device = torch.device("cuda")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    for seed in range(2):
        t1 = time.time()

        set_seed(seed)

        outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=128, do_sample=True, top_p=0.9, num_return_sequences=1,
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
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--load_finetuned", action="store_true")
    
    base_model = "meta-llama/Llama-2-7b-chat-hf"

    args = parser.parse_args()

    saved_path = args.saved_path

    # print("saved path is ", saved_path)

    # config = PeftConfig.from_pretrained(saved_path)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # print("base model name is ", config.base_model_name_or_path)
    
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

    # print("Model before peft loading")
    # print(model)
    # print()

    # Load the LoRA model
    if args.load_finetuned:
        inference_model = PeftModel.from_pretrained(model, saved_path)
    # print("Inference model is: ")
    # print(inference_model)
    # print()

    # print("Model after inplace loading is ")
    # print(model)
    # print()

    # generate_pipeline()
    # generate_generate()

    #prompt = "<s>[INST] <<SYS>>\nYou are a user commenting on the politics domain\n<</SYS>>\n\nGenerate a negative sentiment sentence inside curly braces [/INST]"
    #prompt = "<s>[INST] <<SYS>>\nYou are a user commenting on the politics domain\n<</SYS>>\n\n{} sentiment sentence is: [/INST]"
    #prompt = "<s>[INST] <<SYS>>\nYou are a user commenting on the politics domain, just give the answer without telling anything else\n<</SYS>>\n\n{} sentiment sentence is: [/INST]"
    #prompt = "<s>[INST] <<SYS>>\nYou are a user commenting on the politics domain, just give the answer without telling anything else\n<</SYS>>\n\nPlease generate a sentence indicating a {} sentiment.\nSentence: [/INST]"
    # prompt = '<s>[INST] <<SYS>>\nYou are a user who likes commenting on the politics news\n<</SYS>>\n\nPlease generate a single sentence indicating a {} sentiment as in the example below:\nSentence: "Today the weather is very good."\nSentence: [/INST]'
    prompt = "<s>[INST] <<SYS>>\nYour job is to comment on news from politics.\n<</SYS>>\n\nPlease share a review in one sentence expressing a positive sentence. Please only share the review sentence without any additional response before or after. Please make sure the review is a meaningful sentence.[/INST]"

    # prompt = "<s>[INST] <<SYS>>\nYou are a user commenting on the politics domain\n<</SYS>>\n\nGenerate a negative sentiment sentence [/INST]"
    # generate_text([prompt])

    # prompt = "<s>[INST] <<SYS>>\nYou are a user commenting on the politics domain\n<</SYS>>\n\nGenerate a positive sentiment sentence [/INST]"
    # generate_text([prompt])

    # generate_text([prompt.format("Negative")])
    # generate_text([prompt.format("positive")])
    # generate_text([prompt.format("Neutral")])
    # exit()
    # generate_text(["contradiction \n"])
    # generate_text(["neutral \n"])

    if args.task == "bnsentiment":
        generate_labelaware_bnsentiment()
    elif args.task == "diverse_bnsentiment":
        generate_domainaware_bnsentiment()
    elif args.task == "xnli":
        generate_labelaware(args.seed, args.size)
    elif args.task == "mlheadline":
        generate_labelaware_mlheadline()
    elif args.task == "hiproduct":
        generate_labelaware_hiproduct()

    

if __name__ == "__main__":
    main()