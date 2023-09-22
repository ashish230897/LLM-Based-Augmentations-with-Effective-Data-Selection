import os
import pandas as pd
import argparse

repo_path = os.getcwd()

def form_data(seed, size, lang):
    data_path = repo_path + "/data/xnli/pretraining/processed_{}seed{}size{}.csv".format(lang, seed, size)
    df = pd.read_csv(data_path)
    premises, hypos = list(df["Premises"]), list(df["Hypothesis"])

    file = open(repo_path + "/data/xnli/pretraining/pretrain_{}_seed{}size{}.txt".format(lang, seed, size), "w+")
    for i,pre in enumerate(premises):
        file.write(pre + " " + hypos[i] + "\n")
    file.close()

def form_data_bnsentiment(path):
    
    df = pd.read_csv(path)
    texts = list(df["Texts"])

    file = open(path + ".pretrain", "w+")
    for text in texts:
        file.write(text.strip().replace("\n", "") + "\n")
    file.close()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        type=str,
        required=True
    )

    parser.add_argument(
        "--path",
        type=str,
        required=False
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

    args = parser.parse_args()

    if args.task == "xnli":
        form_data(args.seed, args.size, args.lang)
    else:
        form_data_bnsentiment(args.path)





if __name__ == "__main__":
    main()