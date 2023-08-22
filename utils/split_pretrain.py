import os
import random
import argparse

repo_path = os.getcwd()

def preprocess(seed, lang):
    data_path = repo_path + "/data/xnli/pretraining/pretrain_target_{}_seed{}.txt".format(lang, seed)
    
    file = open(data_path)
    lines = list(file.readlines())

    random.seed(42)
    random.shuffle(lines)

    train = lines[0:int(0.9*len(lines))]
    valid = lines[int(0.9*len(lines)):]
    
    file = open(repo_path + "/data/xnli/pretraining/pretrain.target.{}.seed{}train".format(lang, seed), "w+")
    for line in train:
        file.write(line.strip().replace("\n", "") + "\n")
    file.close()

    file = open(repo_path + "/data/xnli/pretraining/pretrain.target.{}.seed{}valid".format(lang, seed), "w+")
    for line in valid:
        file.write(line.strip().replace("\n", "") + "\n")
    file.close()

    print(len(train), len(valid))


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lang", type=str, required=True)

    args = parser.parse_args()

    preprocess(args.seed, args.lang)


if __name__ == "__main__":
    main()