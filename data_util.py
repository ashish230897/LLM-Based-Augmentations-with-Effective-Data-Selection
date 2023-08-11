# select a random subset of xnli train set

import pandas as pd
import random

repo_path = "/raid/speech/ashish/TSTG_new"

def save_random_subset(premises, hypos, labels, seed, subset_size):
    entail_premises, entail_hypos = [], []
    neut_premises, neut_hypos = [], []
    contra_premises, contra_hypos = [], []

    for i, label in enumerate(labels):
        if label == "neutral":
            neut_premises.append(premises[i])
            neut_hypos.append(hypos[i])
        elif label == "entailment":
            entail_premises.append(premises[i])
            entail_hypos.append(hypos[i])
        else:
            contra_premises.append(premises[i])
            contra_hypos.append(hypos[i])
    
    indices = [i for i in range(len(entail_premises))]
    random.seed(seed)
    entail_indices = random.sample(indices, subset_size)
    entail_premises_subset = [entail_premises[i] for i in entail_indices]
    entail_hypos_subset = [entail_hypos[i] for i in entail_indices]

    indices = [i for i in range(len(neut_premises))]
    random.seed(seed)
    neut_indices = random.sample(indices, subset_size)
    neut_premises_subset = [neut_premises[i] for i in neut_indices]
    neut_hypos_subset = [neut_hypos[i] for i in neut_indices]

    indices = [i for i in range(len(contra_premises))]
    random.seed(seed)
    contra_indices = random.sample(indices, subset_size)
    contra_premises_subset = [contra_premises[i] for i in contra_indices]
    contra_hypos_subset = [contra_hypos[i] for i in contra_indices]
    
    final_text = []
    for i in range(subset_size):
        final_text.append(entail_premises_subset[i].strip().replace("\n", "") + "\t" + entail_hypos_subset[i].strip().replace("\n", "") + "\t" + "entailment")
        final_text.append(neut_premises_subset[i].strip().replace("\n", "") + "\t" + neut_hypos_subset[i].strip().replace("\n", "") + "\t" + "neutral")
        final_text.append(contra_premises_subset[i].strip().replace("\n", "") + "\t" + contra_hypos_subset[i].strip().replace("\n", "") + "\t" + "contradiction")

    file = open("{}/data/xnli/xnli_train_subset_seed{}_size{}.tsv".format(repo_path, seed, subset_size), "w+")
    for text in final_text:
        file.write(text + "\n")
    file.close()



def main():
    df = pd.read_csv("{}/data/xnli/train-en.tsv".format(repo_path), delimiter="\t", names=["premise", "hypothesis", "label"])
    premises, hypos, labels = list(df["premise"]), list(df["hypothesis"]), list(df["label"])

    save_random_subset(premises, hypos, labels, 42, 1500)



if __name__ == "__main__":
    main()