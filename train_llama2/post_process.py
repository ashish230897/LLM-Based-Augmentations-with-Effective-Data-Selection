import pandas as pd
import os
import argparse

path = os.getcwd()

def sst5_postprocess(args):

    df = pd.read_csv(args.input_path)
    texts = list(df["Texts"])

    new_texts = []
    for text in texts:
        if '"' in text:
            new_texts.append(text.split('"')[1])
        else:
            text = text.split('[/INST]')[1]
            if ":" in text:
                new_texts.append(text.split(":")[1])
            else:
                new_texts.append(text)
    
    dict = {"Texts": new_texts}
    df_new = pd.DataFrame(dict)
    df_new.to_csv(args.output_path, index=False)

def bnsentiment_postprocess(args):

    df = pd.read_csv(args.input_path)
    texts = list(df["Texts"])

    new_texts = []
    for text in texts:
        text = text.split('[/INST]')[1]
        if ":" in text:
            text = text.split(":")[-1]

        if '"' in text:
            text = text.split('"')[1]
        
        new_texts.append(text)
    
    texts_v2 = []
    for text in new_texts:
        if text.startswith("Review"):
            texts_v2.append(text[len("Review"):])
        else:
            texts_v2.append(text)


    dict = {"Texts": texts_v2}
    df_new = pd.DataFrame(dict)
    df_new.to_csv(args.output_path, index=False)


def bnsentiment_postprocesslabeled(args):

    df = pd.read_csv(args.input_path)
    texts = list(df["Texts"])

    labels = []


    new_texts = []
    for i,text in enumerate(texts):
        if i%3 == 0:
            labels.append("positive")
        elif i%3 == 1:
            labels.append("negative")
        elif i%3 == 2:
            labels.append("neutral")
        
        text = text.split('[/INST]')[1]
        if ":" in text:
            text = text.split(":")[-1]

        if '"' in text:
            text = text.split('"')[1]
        
        new_texts.append(text)
    
    texts_v2 = []
    for text in new_texts:
        if text.startswith("Review"):
            texts_v2.append(text[len("Review"):])
        else:
            texts_v2.append(text)

    assert len(texts_v2) == len(labels) == len(texts)

    dict = {"Texts": texts_v2, "Labels": labels}
    df_new = pd.DataFrame(dict)
    df_new.to_csv(args.output_path, index=False)

def postprocess_labeled(args):

    df = pd.read_csv(args.input_path)
    texts, labels = list(df["Texts"]), list(df["Labels"])

    new_texts = []
    
    for text in texts:
        
        text = text.split('[/INST]')[1]
        if ":" in text:
            text = text.split(":")[-1]

        if '"' in text:
            text = text.split('"')[1]
        
        new_texts.append(text)
    
    texts_v2 = []
    for text in new_texts:
        if text.startswith("Review"):
            texts_v2.append(text[len("Review"):])
        else:
            texts_v2.append(text)

    assert len(texts_v2) == len(labels) == len(texts)

    dict = {"Texts": texts_v2, "Labels": labels}
    df_new = pd.DataFrame(dict)
    df_new.to_csv(args.output_path, index=False)


def xnli_postprocess(args):

    df = pd.read_csv(args.input_path)
    texts = list(df["Texts"])

    hypos = []
    for text in texts:
        hypo_text = text.split('[/INST]')[1]

        if ":" in hypo_text:
            hypos.append(hypo_text.split(":")[1])
        else:
            hypos.append(hypo_text)
    
    df = pd.read_csv(args.premises_input_path)
    premises = list(df["Texts"])
    premises_triple = []
    for pre in premises:
        premises_triple += [pre]*3

    print(len(premises_triple), len(hypos))

    assert len(premises_triple) == len(hypos)

    
    dict = {"Premises": premises_triple, "Hypothesis": hypos}
    df_new = pd.DataFrame(dict)
    df_new.to_csv(args.output_path, index=False)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, type=str)
    parser.add_argument("--premises_input_path", required=False, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--task", required=True, type=str)
    args = parser.parse_args()


    if args.task == "sst5":
        sst5_postprocess(args)
    elif args.task == "xnlihypos":
        xnli_postprocess(args)
    elif args.task == "bnsentiment" or args.task == "hiproduct" or args.task == "xnli":
        bnsentiment_postprocess(args)
    elif args.task == "hiproductunlabeled":
        bnsentiment_postprocesslabeled(args)
    elif args.task == "hiproductLabeled":
        postprocess_labeled(args)



if __name__ == "__main__":
    main()

# commands
# for processing file that has labels in it:
# python train_llama2/post_process.py --input_path /raid/speech/ashish/TSTG_new/results/generations_zeroshot_marsentiment.csv --output_path /raid/speech/ashish/TSTG_new/results/generations_zeroshot_marsentiment_proc.csv --task hiproductLabeled

# for processing file that do not have labels:
# python train_llama2/post_process.py --input_path /raid/speech/ashish/TSTG_new/results/generations_zeroshot_marsentiment.csv --output_path /raid/speech/ashish/TSTG_new/results/generations_zeroshot_marsentiment_proc.csv --task hiproduct

# for postprocessing xnli premises, hypos:
# python train_llama2/post_process.py --input_path /raid/speech/ashish/TSTG_new/results/generations_zeroshot_nlihypos.csv --output_path /raid/speech/ashish/TSTG_new/results/generations_zeroshot_nliprehypos.csv --premises_input_path /raid/speech/ashish/TSTG_new/data/xnli/pretraining/pretrain_label_enpremises.csv --task xnlihypos