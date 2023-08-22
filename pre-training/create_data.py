import random
from transformers import AutoModelForMaskedLM, AutoTokenizer
import os
import argparse

repo_path = os.getcwd()

def get_random_subset(seed, lang, model):
    random.seed(42)

    file = open(repo_path + "/data/xnli/pretraining/pretrain.target.{}.seed{}train".format(lang, seed))
    train_lines = file.readlines()
    file.close()

    file = open(repo_path + "/data/xnli/pretraining/pretrain.target.{}.seed{}valid".format(lang, seed))
    valid_lines = file.readlines()
    file.close()

    train_lines = [line.split("\t")[0].replace("\n", "") for line in train_lines]
    valid_lines = [line.split("\t")[0].replace("\n", "") for line in valid_lines]


    outfile_train = open(repo_path + "/data/xnli/pretraining/pretrain.target.{}.proc.seed{}train".format(lang, seed), "w+")
    outfile_eval = open(repo_path + "/data/xnli/pretraining/pretrain.target.{}.proc.seed{}valid".format(lang, seed), "w+")

    tokenizer = AutoTokenizer.from_pretrained(model, model_max_length=512, truncation=True)
    
    print("started tokenizing")
    for line in train_lines: 
        tokens = tokenizer.tokenize(line.strip()) 
        # print(tokens) 
        count = len(tokenizer.tokenize(line.strip())) 
        msk_string = line.strip()+'\t' 
        for i in range(count): 
            msk_string += "MASK " 
        msk_string = msk_string.strip() + "\n" 
        outfile_train.write(msk_string)

    for line in valid_lines: 
        tokens = tokenizer.tokenize(line.strip()) 
        # print(tokens) 
        count = len(tokenizer.tokenize(line.strip())) 
        msk_string = line.strip()+'\t' 
        for i in range(count): 
            msk_string += "MASK " 
        msk_string = msk_string.strip() + "\n" 
        outfile_eval.write(msk_string)
    
    outfile_eval.close()
    outfile_train.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)

    args = parser.parse_args()

    get_random_subset(args.seed, args.lang, args.model)

if __name__ == "__main__":
    main()
