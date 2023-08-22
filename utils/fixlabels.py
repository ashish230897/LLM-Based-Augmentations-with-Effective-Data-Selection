import os

repo_path = os.getcwd()

def replace_labels(lang):
    data_path = repo_path + "/data/xnli/test-{}.tsv".format(lang)
    premises, hypos, labels = [], [], []

    with open(data_path) as file:
        for line in file:
            premises.append(line.split('\t')[0])
            hypos.append(line.split('\t')[1])
            labels.append(line.split('\t')[2].replace("\n", ""))

    label_map = {'0':"entailment", '1':"neutral", '2':"contradiction"}

    file = open(data_path, "w+")
    for i,pre in enumerate(premises):
        file.write(pre + "\t" + hypos[i] + "\t" + label_map[labels[i]] + "\n")
    file.close()

def main():
    replace_labels("hi")
    replace_labels("mr")




if __name__ == "__main__":
    main()