from inference.engine import Model
import pandas as pd
import time
from tqdm import tqdm
import argparse
import os

repo_path = os.getcwd() 
ckpt_dir = "{}/IndicTrans2/translations/en-indic-preprint/fairseq_model/".format(repo_path)

model = Model(ckpt_dir, model_type="fairseq")


def translate_synthetic_data(lang, seed, size):
    data_path = "{}/data/xnli/pretraining/processed_seed{}size{}.csv".format(repo_path, seed, size)
    premises, hypos = [], []

    df = pd.read_csv(data_path)
    premises, hypos = list(df["Premises"]), list(df["Hypothesis"])
    
    print(len(premises), len(hypos))

    lang_code_dict = {"hi":"hin_Deva", "en":"eng_Latn", "ur": "urd_Arab", "mr":"mar_Deva"}
    batch_size = 400
    num_batches = int(len(premises)/batch_size)
    print("Number of batches are {}".format(num_batches))

    
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

    dict = {"Premises": premises_txns, "Hypothesis": hypos_txns}
    df = pd.DataFrame(dict)
    df.to_csv("{}/data/xnli/pretraining/processed_{}seed{}size{}.csv".format(repo_path, lang, seed, size), index=False)

    print("Time taken for lang {} is {}".format(lang, time.time() - t1))

def translate_synthetic_bnsentiment(input_path, output_path, lang):
    data_path = input_path
    texts = []

    df = pd.read_csv(data_path,sep='\t')
    # df = pd.read_csv(data_path)

    texts = list(df["Texts"])
    
    print(len(texts))

    lang_code_dict = {"en":"eng_Latn", "bn": "ben_Beng", "ml": "mal_Mlym", "hi": "hin_Deva"}
    batch_size = 1024
    num_batches = int(len(texts)/batch_size)
    print("Number of batches are {}".format(num_batches))

    
    t1 = time.time()
    
    # translate texts
    texts_txns = []
    for i in tqdm(range(num_batches)):
        if i == num_batches-1:
            curr_batch = texts[i*batch_size:]
        else:
            curr_batch = texts[i*batch_size:i*batch_size+batch_size]

        try:            
            curr_translations = model.batch_translate(curr_batch, "eng_Latn", lang_code_dict[lang])
            print("It was set")
        except Exception as e:
            print("Error is ", e)
        
        texts_txns += curr_translations
    
    assert len(texts) == len(texts_txns)

    dict = {"Texts": texts_txns}
    df = pd.DataFrame(dict)
    df.to_csv(output_path, index=False)

    print("Time taken for lang {} is {}".format(lang, time.time() - t1))

def translate_synthetic_labeled(input_path, output_path, lang):
    data_path = input_path

    df = pd.read_csv(data_path)

    texts = list(df["Texts"])
    labels = list(df["Labels"])
    
    print(len(texts))

    lang_code_dict = {"en":"eng_Latn", "bn": "ben_Beng", "ml": "mal_Mlym", "hi": "hin_Deva", "mar": "mar_Deva"}
    batch_size = 80
    num_batches = int(len(texts)/batch_size)
    print("Number of batches are {}".format(num_batches))

    
    t1 = time.time()
    
    # translate texts
    texts_txns = []
    for i in tqdm(range(num_batches)):
        if i == num_batches-1:
            curr_batch = texts[i*batch_size:]
        else:
            curr_batch = texts[i*batch_size:i*batch_size+batch_size]

        try:            
            curr_translations = model.batch_translate(curr_batch, "eng_Latn", lang_code_dict[lang])
            #print("It was set")
        except Exception as e:
            print("Error is ", e)
        
        texts_txns += curr_translations
    
    assert len(texts) == len(texts_txns)

    dict = {"Texts": texts_txns, "Labels": labels}
    df = pd.DataFrame(dict)
    df.to_csv(output_path, index=False)

    print("Time taken for lang {} is {}".format(lang, time.time() - t1))

def translate_synthetic_labeled_nli(input_path, output_path, lang):
    data_path = input_path

    df = pd.read_csv(data_path)

    firsts, seconds = list(df["Premises"]), list(df["Hypothesis"])
    labels = list(df["Label"])
    
    print(len(firsts))

    lang_code_dict = {"en":"eng_Latn", "bn": "ben_Beng", "ml": "mal_Mlym", "hi": "hin_Deva", "mar": "mar_Deva"}
    batch_size = 50
    num_batches = int(len(firsts)/batch_size)
    print("Number of batches are {}".format(num_batches))

    
    t1 = time.time()
    
    # translate texts
    firsts_txns, seconds_txns = [], []
    for i in tqdm(range(num_batches)):
        if i == num_batches-1:
            curr_batch_firsts = firsts[i*batch_size:]
            curr_batch_seconds = seconds[i*batch_size:]
        else:
            curr_batch_firsts = firsts[i*batch_size:i*batch_size+batch_size]
            curr_batch_seconds = seconds[i*batch_size:i*batch_size+batch_size]

        try:            
            curr_translations_firsts = model.batch_translate(curr_batch_firsts, "eng_Latn", lang_code_dict[lang])
            curr_translations_seconds = model.batch_translate(curr_batch_seconds, "eng_Latn", lang_code_dict[lang])
            #print("It was set")
        except Exception as e:
            print("Error is ", e)
        
        firsts_txns += curr_translations_firsts
        seconds_txns += curr_translations_seconds
    
    assert len(firsts) == len(seconds_txns)

    dict = {"Premises": firsts_txns, "Hypothesis": seconds_txns, "Label": labels}
    df = pd.DataFrame(dict)
    df.to_csv(output_path, index=False)

    print("Time taken for lang {} is {}".format(lang, time.time() - t1))

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, required=False)
    parser.add_argument("--size", type=int, required=False)
    parser.add_argument("--lang", type=str, required=False)
    parser.add_argument("--input_path", type=str, required=False)
    parser.add_argument("--output_path", type=str, required=False)

    parser.add_argument("--task", type=str, required=True)

    args = parser.parse_args()

    if args.task == "xnli":
        translate_synthetic_data(args.lang, args.seed, args.size)
    elif args.task == "snli":
        translate_synthetic_labeled_nli(args.input_path, args.output_path, args.lang)
    elif args.task == "hiproductlabeled":
        translate_synthetic_labeled(args.input_path, args.output_path, args.lang)
    else:
        translate_synthetic_bnsentiment(args.input_path, args.output_path, args.lang)


if __name__ == "__main__":
    main()

#CUDA_VISIBLE_DEVICES=5 python IndicTrans2/translate_synthetic.py --input_path "data/hiproduct/pretraining/pretrain_few-shot.en" --output_path "data/hiproduct/pretraining/pretrain_few-shot.hi"  --lang hi --task bnsentiment
# for labeled data:
# python IndicTrans2/translate_synthetic.py --input_path /raid/speech/ashish/TSTG_new/data/marsentiment/pretraining/pretrain_label_en.csv --output_path /raid/speech/ashish/TSTG_new/data/marsentiment/pretraining/pretrain_label_mar.csv  --lang hi --task hiproductlabeled

# CUDA_VISIBLE_DEVICES=0 python IndicTrans2/translate_synthetic.py --input_path "data/hiprod-new/zero-shot-gen-hiprod.en" --output_path "data/hiprod-new/zero-shot-gen-hiprod.hi"  --lang hi --task bnsentiment