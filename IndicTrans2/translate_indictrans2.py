from inference.engine import Model
import pandas as pd
import time
import argparse
import os
from tqdm import tqdm

repo_path = os.getcwd() 
ckpt_dir = "{}/IndicTrans2/translations/en-indic-preprint/fairseq_model/".format(repo_path)

model = Model(ckpt_dir, model_type="fairseq")


def translate_xnli_english_subset(seed, size, lang):
    data_path = "{}/data/xnli/train-ensubsetseed{}size{}.tsv".format(repo_path, seed, size)
    premises, hypos, labels = [], [], []
    
    with open(data_path) as file:
        for line in file:
            premises.append(line.split("\t")[0])
            hypos.append(line.split("\t")[1])
            labels.append(line.split("\t")[2].replace("\n", ""))
    
    print(len(premises), len(hypos), len(labels))

    lang_code_dict = {"hi":"hin_Deva", "en":"eng_Latn", "ur": "urd_Arab", "mr":"mar_Deva"}
    langs = [lang]
    batch_size = 200
    num_batches = int(len(premises)/batch_size)

    for lang in langs:
        t1 = time.time()
        # translate premises
        premises_txns = []
        for i in tqdm(range(num_batches)):
            if i == num_batches-1:
                curr_batch = premises[i*batch_size:]
            else:
                curr_batch = premises[i*batch_size:i*batch_size+batch_size]

            try:            
                curr_translations = model.batch_translate(curr_batch, "eng_Latn", lang_code_dict[lang])
            except Exception as e:
                print("Error is ", e)
            
            
            premises_txns += curr_translations
        
        assert len(premises) == len(premises_txns)


        # translate hypothesis
        hypos_txns = []
        for i in tqdm(range(num_batches)):
            if i == num_batches-1:
                curr_batch = hypos[i*batch_size:]
            else:
                curr_batch = hypos[i*batch_size:i*batch_size+batch_size]

            try:
                curr_translations = model.batch_translate(curr_batch, "eng_Latn", lang_code_dict[lang])
            except Exception as e:
                print("Error is ", e)
            
            hypos_txns += curr_translations
        
        assert len(hypos) == len(hypos_txns)

        file = open("{}/data/xnli/train-{}subsetseed{}size{}.tsv".format(repo_path, lang, seed, size), "w+")
        for i in range(len(hypos)):
            file.write(premises_txns[i] + "\t" + hypos_txns[i] + "\t" + labels[i].replace("\n", "") + "\n")
        file.close()

        print("Time taken for lang {} is {}".format(lang, time.time() - t1))

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

    parser.add_argument(
        "--lang",
        type=str,
        required=True
    )

    args = parser.parse_args()

    translate_xnli_english_subset(args.seed, args.size, args.lang)


if __name__ == "__main__":
    main()