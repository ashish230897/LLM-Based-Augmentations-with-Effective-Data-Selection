import os
import random
import argparse

repo_path = os.getcwd()

def preprocess(file_path):
    data_path = file_path
    
    file = open(data_path)
    lines = list(file.readlines())

    random.seed(42)
    random.shuffle(lines)

    # train = lines[0:int(0.9*len(lines))]
    # valid = lines[int(0.9*len(lines)):]
    train = lines[0:130000]
    valid = lines[130000:140000]

    if len(valid) < 10000:
        train = lines[0:int(0.9*len(lines))]
        valid = lines[int(0.9*len(lines)):]

    
    file = open(file_path + ".train", "w+")
    for line in train:
        file.write(line.strip().replace("\n", "") + "\n")
    file.close()

    file = open(file_path + ".valid", "w+")
    for line in valid:
        file.write(line.strip().replace("\n", "") + "\n")
    file.close()

    print(len(train), len(valid))


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--file_path", type=str, required=True)

    args = parser.parse_args()

    preprocess(args.file_path)


if __name__ == "__main__":
    main()