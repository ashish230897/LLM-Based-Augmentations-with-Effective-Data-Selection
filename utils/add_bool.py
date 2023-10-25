import csv
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_path", type=str, required=True)
    parser.add_argument("--train_file_path", type=str, required=True)
    parser.add_argument("--output_file_path", type=str, required=True)

    args = parser.parse_args()

    file = open(args.gen_path)
    lines = file.readlines()
    file.close()
    
    new_gens = []
    for line in lines:
        new_gens.append(line.strip().replace("\n", "") + "\t" + "neutral" + "\t" + "0\n")
    
    file = open(args.train_file_path)
    lines = file.readlines()
    file.close()

    for line in lines:
        if len(line.split("\t")) > 2:
            new_gens.append(line.split("\t")[0].strip().replace("\n", "") + "\t" + line.split("\t")[1].strip().replace("\n", "") + "\t" + line.split("\t")[2].strip().replace("\n", "") + "\t" + "1\n")
        else:
            new_gens.append(line.split("\t")[0].strip().replace("\n", "") + "\t" + line.split("\t")[1].strip().replace("\n", "") + "\t" + "1\n")
    
    file = open(args.output_file_path, "w+")
    for gen in new_gens:
        file.write(gen)
    file.close()

if __name__ == "__main__":
    main()

# command
# python utils/add_bool.py --gen_path /raid/speech/ashish/TSTG_new/data/bnsentiment/zero-shot_ST_v2_topk/expt-3/top_2.5k_each.en --train_file_path /raid/speech/ashish/TSTG_new/data/sst5/train-en.tsv --output_file_path /raid/speech/ashish/TSTG_new/data/bnsentiment/zero-shot_ST_v2_topk/expt-3/train-en.tsv