import pandas as pd
import os

repo_path = '/raid/speech/ashish/TSTG_new'

def convert_text(df):
    texts = []
    
    for _, row in df.iterrows():
        premise, hypo = row["premise"], row["hypothesis"]
        text = "Premise: {} \nHypothesis: {}".format(premise, hypo)
        texts.append(text)
    
    df["text"] = texts
    
    return df


def main():
    df = pd.read_csv("{}/data/xnli/xnli_train_subset_seed42_size1500.tsv".format(repo_path), delimiter="\t", names=["premise", "hypothesis", "label"])
    df = convert_text(df)

    print("First row is:")
    print(df.iloc[0])

    df.to_csv("{}/data/xnli/xnli_train_subset_seed42_size1500_texts.csv".format(repo_path), index=False)

if __name__ == "__main__":
    main()

