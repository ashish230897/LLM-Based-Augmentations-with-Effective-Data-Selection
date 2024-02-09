import random

def get_topks():
    lang = "en"
    expt = "3"

    file = open("./data/marsentiment/diverseK/topk/7.5k/train-{}.tsv".format(lang))
    lines = file.readlines()
    file.close()

    neutrals = lines[0:7500]
    positives = lines[7500:15000]
    negatives = lines[15000:22500]

    train_lines = lines[22500:]


    file = open("./data/marsentiment/diverseK/topk/3.5k/train-{}.tsv".format(lang), "w+")
    for sent in neutrals[0:3500]:
        file.write(sent)
    for sent in positives[0:3500]:
        file.write(sent)
    for sent in negatives[0:3500]:
        file.write(sent)
    for sent in train_lines:
        file.write(sent)
    file.close()


    file = open("./data/marsentiment/diverseK/topk/4.5k/train-{}.tsv".format(lang), "w+")
    for sent in neutrals[0:4500]:
        file.write(sent)
    for sent in positives[0:4500]:
        file.write(sent)
    for sent in negatives[0:4500]:
        file.write(sent)
    for sent in train_lines:
        file.write(sent)
    file.close()


    file = open("./data/marsentiment/diverseK/topk/5.5k/train-{}.tsv".format(lang), "w+")
    for sent in neutrals[0:5500]:
        file.write(sent)
    for sent in positives[0:5500]:
        file.write(sent)
    for sent in negatives[0:5500]:
        file.write(sent)
    for sent in train_lines:
        file.write(sent)
    file.close()
    
    file = open("./data/marsentiment/diverseK/topk/6.5k/train-{}.tsv".format(lang), "w+")
    for sent in neutrals[0:6500]:
        file.write(sent)
    for sent in positives[0:6500]:
        file.write(sent)
    for sent in negatives[0:6500]:
        file.write(sent)
    for sent in train_lines:
        file.write(sent)
    file.close()

def get_randpropmtks():
    
    expt, lang = 3, "en"
    
    file = open("./data/marsentiment/pretraining/pretrain_label_{}.csv.labeledpretrain.train".format(lang))
    lines = file.readlines()
    file.close()
    texts, labels = [line.split("\t")[0].strip() for line in lines], [line.split("\t")[1].strip().replace("\n", "") for line in lines]
    
    neutrals, positives, negatives = [], [], []
    for text, label in zip(texts, labels):
        if label == "neutral": neutrals.append(text)
        elif label == "negative": negatives.append(text)
        elif label == "positive": positives.append(text)
        else: print("label not identified", label)
    
    print(len(neutrals), len(positives), len(negatives))
    
    random.seed(42)
    random.shuffle(neutrals)
    random.shuffle(positives)
    random.shuffle(negatives)
    
    file = open("./data/marsentiment/zero-shot_topk/expt-{}/7.5k/train-{}.tsv".format(expt, lang))
    lines = file.readlines()
    file.close()
    train_lines = lines[22500:]
    
    file = open("./data/marsentiment/rand_prompt/expt-{}/2.5k/train-{}.tsv".format(expt, lang), "w+")
    for sent in neutrals[0:2500]:
        file.write(sent + "\t" + "neutral" + "\t" + "1\n")
    for sent in positives[0:2500]:
        file.write(sent + "\t" + "positive" + "\t" + "1\n")
    for sent in negatives[0:2500]:
        file.write(sent + "\t" + "negative" + "\t" + "1\n")
    for sent in train_lines:
        file.write(sent)
    file.close()
    
    file = open("./data/marsentiment/rand_prompt/expt-{}/3.5k/train-{}.tsv".format(expt, lang), "w+")
    for sent in neutrals[0:3500]:
        file.write(sent + "\t" + "neutral" + "\t" + "1\n")
    for sent in positives[0:3500]:
        file.write(sent + "\t" + "positive" + "\t" + "1\n")
    for sent in negatives[0:3500]:
        file.write(sent + "\t" + "negative" + "\t" + "1\n")
    for sent in train_lines:
        file.write(sent)
    file.close()
    
    file = open("./data/marsentiment/rand_prompt/expt-{}/4.5k/train-{}.tsv".format(expt, lang), "w+")
    for sent in neutrals[0:4500]:
        file.write(sent + "\t" + "neutral" + "\t" + "1\n")
    for sent in positives[0:4500]:
        file.write(sent + "\t" + "positive" + "\t" + "1\n")
    for sent in negatives[0:4500]:
        file.write(sent + "\t" + "negative" + "\t" + "1\n")
    for sent in train_lines:
        file.write(sent)
    file.close()
    
    file = open("./data/marsentiment/rand_prompt/expt-{}/5.5k/train-{}.tsv".format(expt, lang), "w+")
    for sent in neutrals[0:5500]:
        file.write(sent + "\t" + "neutral" + "\t" + "1\n")
    for sent in positives[0:5500]:
        file.write(sent + "\t" + "positive" + "\t" + "1\n")
    for sent in negatives[0:5500]:
        file.write(sent + "\t" + "negative" + "\t" + "1\n")
    for sent in train_lines:
        file.write(sent)
    file.close()
    
    file = open("./data/marsentiment/rand_prompt/expt-{}/6.5k/train-{}.tsv".format(expt, lang), "w+")
    for sent in neutrals[0:6500]:
        file.write(sent + "\t" + "neutral" + "\t" + "1\n")
    for sent in positives[0:6500]:
        file.write(sent + "\t" + "positive" + "\t" + "1\n")
    for sent in negatives[0:6500]:
        file.write(sent + "\t" + "negative" + "\t" + "1\n")
    for sent in train_lines:
        file.write(sent)
    file.close()


get_topks()
# get_randpropmtks()