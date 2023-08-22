import os
import pandas as pd

repo_path = os.getcwd()

def form_data():
    data_path = repo_path + "/data/xnli/pretraining/processed_mr.csv"
    df = pd.read_csv(data_path)
    premises, hypos = list(df["Premises"]), list(df["Hypothesis"])

    file = open(repo_path + "/data/xnli/pretraining/pretrain_mr.txt", "w+")
    for i,pre in enumerate(premises):
        file.write(pre + " " + hypos[i] + "\n")
    file.close()

def main():
    form_data()




if __name__ == "__main__":
    main()