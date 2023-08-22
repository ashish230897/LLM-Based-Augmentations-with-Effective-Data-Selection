import pandas as pd
import os
import argparse

repo_path = '/raid/speech/ashish/TSTG_new'

def convert_text(premises, hypos, labels):
    texts = []
    text_labels = []
    
    i = 0
    for pre, hypo in zip(premises, hypos):
        text = "Premise: {} \nHypothesis: {}".format(pre, hypo)
        texts.append(text)

        text_label = "{} \nPremise: {} \nHypothesis: {}".format(labels[i], pre, hypo)
        text_labels.append(text_label)

        i += 1
    
    print("Lenght of texts is ", len(texts))

    dict = {"Premises": premises, "Hypothesis": hypos, "labels": labels, "text": texts, "text_label": text_labels}
    
    return pd.DataFrame(dict)


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

    args = parser.parse_args()

    data_path = "{}/data/xnli/train-ensubsetseed{}size{}.tsv".format(repo_path, args.seed, args.size)
    premises, hypos, labels = [], [], []

    with open(data_path) as file:
        for line in file:
            premises.append(line.split('\t')[0])
            hypos.append(line.split('\t')[1])
            labels.append(line.split('\t')[2].replace("\n", ""))

    print(len(premises), len(hypos), len(labels))
    
    df = convert_text(premises, hypos, labels)

    print("First row is:")
    print(df.iloc[0])
    print("Length of dataframe is", len(df))

    df.to_csv("{}/data/xnli/train-ensubsetseed{}size{}_texts.csv".format(repo_path, args.seed, args.size), index=False)


if __name__ == "__main__":
    main()

