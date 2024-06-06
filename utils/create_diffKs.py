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


def get_randpropmt_sentiment_sourcedata(pretrain_file_path, source_file_train_path, out_file_path, size):
    
    file = open(pretrain_file_path)
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

    file = open(source_file_train_path)
    train_lines = file.readlines()
    file.close()

    file = open(out_file_path, "w+")
    for sent in neutrals[0:size]:
        file.write(sent + "\t" + "neutral" + "\t" + "1\n")
    for sent in positives[0:size]:
        file.write(sent + "\t" + "positive" + "\t" + "1\n")
    for sent in negatives[0:size]:
        file.write(sent + "\t" + "negative" + "\t" + "1\n")
    for sent in train_lines:
        file.write(sent.replace("\n", "") + "\t" + "1\n")
    file.close()


def get_randpropmt_nli_sourcedata(pretrain_file_path, source_file_train_path, out_file_path, size):
    
    file = open(pretrain_file_path)
    lines = file.readlines()
    file.close()
    texts, labels = [line.split("\t")[0].strip() + "\t" + line.split("\t")[1].strip() for line in lines], [line.split("\t")[2].strip().replace("\n", "") for line in lines]

    neutrals, positives, negatives = [], [], []
    for text, label in zip(texts, labels):
        if label == "neutral": neutrals.append(text)
        elif label == "contradiction": negatives.append(text)
        elif label == "entailment": positives.append(text)
        else: print("label not identified", label)
    
    print(len(neutrals), len(positives), len(negatives))

    random.seed(42)
    random.shuffle(neutrals)
    random.shuffle(positives)
    random.shuffle(negatives)

    file = open(source_file_train_path)
    train_lines = file.readlines()
    file.close()

    file = open(out_file_path, "w+")
    for sent in neutrals[0:size]:
        file.write(sent + "\t" + "neutral" + "\t" + "1\n")
    for sent in positives[0:size]:
        file.write(sent + "\t" + "entailment" + "\t" + "1\n")
    for sent in negatives[0:size]:
        file.write(sent + "\t" + "contradiction" + "\t" + "1\n")
    for sent in train_lines:
        file.write(sent.replace("\n", "") + "\t" + "1\n")
    file.close()

def get_randpropmt_nlidata(pretrain_file_path, out_file_path, size):
    
    file = open(pretrain_file_path)
    lines = file.readlines()
    file.close()
    texts, labels = [line.split("\t")[0].strip() + "\t" + line.split("\t")[1].strip() for line in lines], [line.split("\t")[2].strip().replace("\n", "") for line in lines]

    neutrals, positives, negatives = [], [], []
    for text, label in zip(texts, labels):
        if label == "neutral": neutrals.append(text)
        elif label == "contradiction": negatives.append(text)
        elif label == "entailment": positives.append(text)
        else: print("label not identified", label)
    
    print(len(neutrals), len(positives), len(negatives))

    random.seed(42)
    random.shuffle(neutrals)
    random.shuffle(positives)
    random.shuffle(negatives)

    file = open(source_file_train_path)
    train_lines = file.readlines()
    file.close()

    file = open(out_file_path, "w+")
    for sent in neutrals[0:size]:
        file.write(sent + "\t" + "neutral" + "\t" + "1\n")
    for sent in positives[0:size]:
        file.write(sent + "\t" + "entailment" + "\t" + "1\n")
    for sent in negatives[0:size]:
        file.write(sent + "\t" + "contradiction" + "\t" + "1\n")
    for sent in train_lines:
        file.write(sent.replace("\n", "") + "\t" + "1\n")
    file.close()

# get_randpropmt_sentiment_sourcedata("./data/marsentiment/pretraining/pretrain_label_en.csv.labeledpretrain.train", "./data/sst5/train-en.tsv", 
#                                     "./data/marsentiment/rand_prompt/expt-3/randsourcebl/train-en.tsv")
# get_randpropmt_sentiment_sourcedata("./data/marsentiment/pretraining/pretrain_label_mar.csv.labeledpretrain.train", "./data/sst5/train-mar.tsv", 
#                                     "./data/marsentiment/rand_prompt/expt-2/randsourcebl/train-mar.tsv")
# get_randpropmt_sentiment_sourcedata("./data/hiproduct/pretraining/pretrain_label_en.csv.labeledpretrain.train.wneutral", "./data/sst5/train-en.tsv", 
#                                     "./data/hiproduct/rand_prompt/expt-3/randsourcebl/train-en.tsv")
# get_randpropmt_sentiment_sourcedata("./data/hiproduct/pretraining/pretrain_label_hi.csv.labeledpretrain.train.wneutral", "./data/sst5/train-hi.tsv", 
#                                     "./data/hiproduct/rand_prompt/expt-2/randsourcebl/train-hi.tsv")

# get_randpropmt_sentiment_sourcedata("./data/gluecos/pretraining/pretrain_label_en.csv.labeledpretrain.train", "./data/sst5/train-en.tsv", 
#                                     "./data/gluecos/rand_prompt/expt-3/randsourcebl/train-en.tsv")
# get_randpropmt_sentiment_sourcedata("./data/gluecos/pretraining/pretrain_label_enhid.csv.labeledpretrain.train", "./data/sst5/train-enhid.tsv", 
#                                     "./data/gluecos/rand_prompt/expt-2/randsourcebl/train-enhid.tsv")


# get_randpropmt_nli_sourcedata("./data/snli/pretraining/pretrain_label_enprehypos.csv.labeledpretrain.train", "./data/snli/train-en.tsv", 
#                                     "./data/snli/rand_prompt/expt-3/randsourcebl/train-en.tsv")
# get_randpropmt_nli_sourcedata("./data/snli/pretraining/pretrain_label_hiprehypos.csv.labeledpretrain.train", "./data/snli/train-hi.tsv", 
#                                     "./data/snli/rand_prompt/expt-2/randsourcebl/train-hi.tsv")
# get_randpropmt_nli_sourcedata("./data/snli/pretraining/pretrain_label_urprehypos.csv.labeledpretrain.train", "./data/snli/train-ur.tsv", 
#                                     "./data/snli/rand_prompt/expt-2/randsourcebl/train-ur.tsv")
# get_randpropmt_nli_sourcedata("./data/snli/pretraining/pretrain_label_swprehypos.csv.labeledpretrain.train", "./data/snli/train-sw.tsv", 
#                                     "./data/snli/rand_prompt/expt-2/randsourcebl/train-sw.tsv")
#get_topks()
# get_randpropmtks()

# get_randpropmt_sentiment_sourcedata("./data/marsentiment/pretraining/pretrain_label_mar.csv.labeledpretrain.train", "./data/sst5/train-mar.tsv", 
#                                     "./data/marsentiment/rand_prompt/expt-2/7.5k/train-mar.tsv", 7500)
# get_randpropmt_sentiment_sourcedata("./data/marsentiment/pretraining/pretrain_label_mar.csv.labeledpretrain.train", "./data/sst5/train-mar.tsv", 
#                                     "./data/marsentiment/rand_prompt/expt-2/12.5k/train-mar.tsv", 12500)
# get_randpropmt_sentiment_sourcedata("./data/marsentiment/pretraining/pretrain_label_mar.csv.labeledpretrain.train", "./data/sst5/train-mar.tsv", 
#                                     "./data/marsentiment/rand_prompt/expt-2/17.5k/train-mar.tsv", 17500)
# get_randpropmt_sentiment_sourcedata("./data/marsentiment/pretraining/pretrain_label_mar.csv.labeledpretrain.train", "./data/sst5/train-mar.tsv", 
#                                     "./data/marsentiment/rand_prompt/expt-2/22.5k/train-mar.tsv", 22500)

get_randpropmt_nli_sourcedata("./data/snli/pretraining/pretrain_label_hiprehypos.csv.labeledpretrain.train", "./data/xnli/train-hi.tsv", 
                                    "./data/xnli/rand_prompt/expt-2/5k/train-hi.tsv", 5000)
get_randpropmt_nli_sourcedata("./data/snli/pretraining/pretrain_label_urprehypos.csv.labeledpretrain.train", "./data/xnli/train-ur.tsv", 
                                    "./data/xnli/rand_prompt/expt-2/5k/train-ur.tsv", 5000)
get_randpropmt_nli_sourcedata("./data/snli/pretraining/pretrain_label_swprehypos.csv.labeledpretrain.train", "./data/xnli/train-sw.tsv", 
                                    "./data/xnli/rand_prompt/expt-2/5k/train-sw.tsv", 5000)