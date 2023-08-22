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

def main():

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

    parser.add_argument(
        "--lang",
        type=str,
        required=True
    )

    args = parser.parse_args()

    form_data(args.lang, args.seed, args.size)




if __name__ == "__main__":
    main()