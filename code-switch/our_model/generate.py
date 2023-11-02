from evaluate import load
import argparse
import pandas as pd

from tqdm import tqdm
import time
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
import os
from peft import PeftModel, PeftConfig

tokenizer = None
model = None

repo_path = os.getcwd()

def inference(inputs, batch_size):

    device = torch.device("cuda")

    num_batches = int(len(inputs)/batch_size)
    print("Number of batches are {}".format(num_batches))

    out_texts = []

    for i in tqdm(range(num_batches)):

        t1 = time.time()

        if i == num_batches-1:
            curr_batch = inputs[i*batch_size:]
        else:
            curr_batch = inputs[i*batch_size:i*batch_size+batch_size]
        

        tokenized_inputs = tokenizer(curr_batch, return_tensors="pt", padding=True).to(device)

        input_ids, attention_mask = tokenized_inputs["input_ids"], tokenized_inputs["attention_mask"]
        set_seed(42)
        
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, min_length=5, max_length=64,
                                num_beams=5, early_stopping=True, no_repeat_ngram_size=2,
                                eos_token_id=tokenizer.eos_token_id, remove_invalid_values=True, num_return_sequences=1)


        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # removing incomplete generations
        for text in texts:
            # if "." in text:
            #     out_texts.append(".".join(text.split(".")[:-1]) + ".")
            # else:
            #     out_texts.append(text)
            out_texts.append(text)
        
        print("Time taken for batch {} is {}".format(i, time.time()-t1))
    
    return out_texts


def main():
    sari = load("sari")
    bleu = load("bleu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=False)
    parser.add_argument("--finetuned_path", type=str, required=False)
    parser.add_argument("--base_model", type=str, required=False)
    parser.add_argument("--batch_size", type=int, required=False)
    parser.add_argument("--out_path", type=str, required=False)
    parser.add_argument("--load_in_8bit", action="store_true")


    args = parser.parse_args()

    print("load in is 8 bit is ", args.load_in_8bit)

    global tokenizer
    global model

    finetuned_path = args.finetuned_path
    #base_model = "bigscience/mt0-large"
    base_model = args.base_model
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model,
        load_in_8bit=args.load_in_8bit,
        trust_remote_code=True,
        use_auth_token="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",
        device_map="auto"
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="right")

    inference_model = PeftModel.from_pretrained(model, finetuned_path)
    model = inference_model.merge_and_unload()  # merges lora parameters into model parameters for faster inference

    print(model)

    file = open(args.file_path)
    lines = file.readlines()
    file.close()

    sents, labels = [], []
    for line in lines:
        sents.append(line.split("\t")[0].replace("\n", ""))
        labels.append(line.split("\t")[1].replace("\n", ""))

    t1 = time.time()
    preds = inference(sents, args.batch_size)
    print("Time taken is ", time.time() - t1)

    assert len(preds) == len(labels)

    file = open(args.out_path, "w+")
    for i,sent in enumerate(preds):
        file.write(sent + "\t" + labels[i] + "\n")
    file.close()



if __name__ == "__main__":
    main()

# python generate.py --base_model bigscience/mt0-xl --batch_size 5 --file_path ./data/valid.csv --finetuned_path ./models/mt0xlv1/checkpoint-800/ --out_path ./data/valid_mt0xlv1.csv