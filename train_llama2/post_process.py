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
    print(len(texts_v2), len(texts), len(labels))

    dict = {"Texts": texts_v2, "Labels": labels}
    df_new = pd.DataFrame(dict)
    df_new.to_csv(args.output_path, index=False)


def xnli_postprocess(args):

    df = pd.read_csv(args.input_path)
    
    texts = list(df["Texts"])
    
    df = pd.read_csv(args.premises_input_path)
    premises = list(df["Texts"])
    labels = ["entailment", "neutral", "contradiction"]

    hypos, new_premises, new_labels = [], [], []
    
    for i,text in enumerate(texts):
        #if "She skied down the mossy meadows and sailed through the calm seas." in text: continue  # ignoring this text
        hypo_text = text.split('[/INST]')[1]

        if ":" in hypo_text:
            hypos.append(hypo_text.split(":")[1].replace("\n", ""))
        else:
            hypos.append(hypo_text.replace("\n", ""))

        new_premises.append(premises[int(i/3)])
        new_labels.append(labels[int(i%3)])
    
    print(len(premises), len(new_premises), len(hypos))

    assert len(new_premises) == len(hypos) == len(new_labels)

    hypo_dict = {}
    for i,hypo in enumerate(hypos):
        hypo_dict[hypo] = [new_premises[i], new_labels[i]]

    #dict = {"Premises": new_premises, "Hypothesis": hypos, "Label": new_labels}
    dict = {"Hypothesis": hypos}
    df_new = pd.DataFrame(dict)
    #print(len(list(df_new["Premises"])))
    
    df_new.to_csv(args.output_path, index=False)

    df = pd.read_csv(args.output_path)
    print(len(df["Hypothesis"]))

    hypos_new = list(df["Hypothesis"])
    premises_new, labels_new, hypos = [], [], []
    for hypo in hypos_new:
        if hypo in hypo_dict:
            premises_new.append(hypo_dict[hypo][0])
            labels_new.append(hypo_dict[hypo][1])
            hypos.append(hypo)
        # else:
        #     premises_new.append("none")
        #     labels_new.append("none")

    dict = {"Premises": premises_new, "Hypothesis": hypos, "Label": labels_new}
    df_new = pd.DataFrame(dict)
    
    df_new.to_csv(args.output_path, index=False)
    df = pd.read_csv(args.output_path)
    print(len(df["Hypothesis"]))




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