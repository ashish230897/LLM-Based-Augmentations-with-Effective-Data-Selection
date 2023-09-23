import os
import pandas as pd
import argparse

repo_path = os.getcwd()

def form_data(lang, seed, size):
    data_path = repo_path + "/data/xnli/train-{}subsetseed{}size{}.tsv".format(lang, seed, size)
    file = open(data_path)
    lines = file.readlines()

    premises, hypos = [], []
    for line in lines:
        premise, hypo = line.split("\t")[0].strip().replace("\n", ""), line.split("\t")[1].strip().replace("\n", "")
        premises.append(premise)
        hypos.append(hypo)

    file = open(repo_path + "/data/xnli/pretraining/pretrain_target_{}_seed{}.txt".format(lang, seed), "w+")
    for i,pre in enumerate(premises):
        file.write(pre + " " + hypos[i] + "\n")
    file.close()

def form_data_clss(input_path):
    data_path = input_path
    file = open(data_path)
    lines = file.readlines()

    texts = []
    for line in lines:
        text = line.split("\t")[0].strip().replace("\n", "")
        texts.append(text)

    file = open(input_path + ".pretrain", "w+")
    for i,text in enumerate(texts):
        file.write(text + "\n")
    file.close()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        type=str,
        required=True
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        required=False
    )

    parser.add_argument(
        "--size",
        type=int,
        required=False
    )

    parser.add_argument(
        "--lang",
        type=str,
        required=False
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=False
    )

    args = parser.parse_args()

    if args.task == "xnli":
        form_data(args.lang, args.seed, args.size)
    elif args.task == "bnsentiment" or args.task == "mlheadline" or args.task == "hiproduct":
        form_data_clss(args.input_path)


if __name__ == "__main__":
    main()