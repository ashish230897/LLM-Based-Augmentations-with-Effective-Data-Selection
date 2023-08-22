import os
import pandas as pd

repo_path = os.getcwd()

def preprocess():
    data_path = repo_path + "/results/generations_label.csv"
    
    df = pd.read_csv(data_path)
    print("Length of dataframe before removing nan rows", len(df))
    
    df.dropna(inplace=True)
    print("Length of dataframe after removing nan rows", len(df))
    
    premises, hypos = list(df["Premises"]), list(df["Hypothesis"])
    texts = []
    for pre,hypo in zip(premises, hypos):
        texts.append(pre + "\t" + hypo)

    # removing duplicates
    texts = list(set(texts))

    # save in a file
    premises, hypos = [], []
    for text in texts:
        pre, hypo = text.split("\t")[0], text.split("\t")[1]
        premises.append(pre)
        hypos.append(hypo)
    
    dict = {"Premises": premises, "Hypothesis": hypos}
    df = pd.DataFrame(dict)
    df.to_csv(repo_path + "/data/xnli/pretraining/processed.csv", index=False)


def main():
    preprocess()


if __name__ == "__main__":
    main()