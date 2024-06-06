import random

def make_snlidata():
    file = open("./data/snli/pretraining/pretrain_label_enprehypos.csv.labeledpretrain.train")
    nli_lines = file.readlines()
    file.close()

    neus, contra, entails = [], [], []
    for line in nli_lines:
        text = line.split("\t")[0].strip() + "\t" + line.split("\t")[1].strip()
        label = line.split("\t")[2].strip().replace("\n", "")
        if label == "entailment": entails.append(text)
        elif label == "neutral": neus.append(text)
        elif label == "contradiction": contra.append(text)

    print(len(neus), len(contra), len(entails))

    assert len(neus) + len(contra) + len(entails) == len(nli_lines)

    random.seed(42)
    random.shuffle(neus)
    random.shuffle(entails)
    random.shuffle(contra)

    train_nli_texts = neus[0:7500] + entails[0:7500] + contra[0:7500]
    train_nli_labels = ["neutral"]*7500 + ["entailment"]*7500 + ["contradiction"]*7500

    dev_nli_texts = neus[7500:8500] + entails[7500:8500] + contra[7500:8500]
    dev_nli_labels = ["neutral"]*1000 + ["entailment"]*1000 + ["contradiction"]*1000

    file = open("./data/snli_promptonly/train-en.tsv", "w+")
    for i in range(22500):
        file.write(train_nli_texts[i] + "\t" + train_nli_labels[i] + "\n")
    file.close()

    file = open("./data/snli_promptonly/dev-en.tsv", "w+")
    for i in range(3000):
        file.write(dev_nli_texts[i] + "\t" + dev_nli_labels[i] + "\n")
    file.close()

    #------------------------------------------------------------------------------------------------------------------

    file = open("./data/snli/pretraining/pretrain_label_hiprehypos.csv.labeledpretrain.train")
    nli_lines = file.readlines()
    file.close()

    neus, contra, entails = [], [], []
    for line in nli_lines:
        text = line.split("\t")[0].strip() + "\t" + line.split("\t")[1].strip()
        label = line.split("\t")[2].strip().replace("\n", "")
        if label == "entailment": entails.append(text)
        elif label == "neutral": neus.append(text)
        elif label == "contradiction": contra.append(text)

    print(len(neus), len(contra), len(entails))

    assert len(neus) + len(contra) + len(entails) == len(nli_lines)

    random.seed(42)
    random.shuffle(neus)
    random.shuffle(entails)
    random.shuffle(contra)

    train_nli_texts = neus[0:7500] + entails[0:7500] + contra[0:7500]
    train_nli_labels = ["neutral"]*7500 + ["entailment"]*7500 + ["contradiction"]*7500

    dev_nli_texts = neus[7500:8500] + entails[7500:8500] + contra[7500:8500]
    dev_nli_labels = ["neutral"]*1000 + ["entailment"]*1000 + ["contradiction"]*1000

    file = open("./data/snli_promptonly/train-hi.tsv", "w+")
    for i in range(22500):
        file.write(train_nli_texts[i] + "\t" + train_nli_labels[i] + "\n")
    file.close()

    file = open("./data/snli_promptonly/dev-hi.tsv", "w+")
    for i in range(3000):
        file.write(dev_nli_texts[i] + "\t" + dev_nli_labels[i] + "\n")
    file.close()
    
    #-------------------------------------------------------------------------------------
    file = open("./data/snli/pretraining/pretrain_label_urprehypos.csv.labeledpretrain.train")
    nli_lines = file.readlines()
    file.close()

    neus, contra, entails = [], [], []
    for line in nli_lines:
        text = line.split("\t")[0].strip() + "\t" + line.split("\t")[1].strip()
        label = line.split("\t")[2].strip().replace("\n", "")
        if label == "entailment": entails.append(text)
        elif label == "neutral": neus.append(text)
        elif label == "contradiction": contra.append(text)

    print(len(neus), len(contra), len(entails))

    assert len(neus) + len(contra) + len(entails) == len(nli_lines)

    random.seed(42)
    random.shuffle(neus)
    random.shuffle(entails)
    random.shuffle(contra)

    train_nli_texts = neus[0:7500] + entails[0:7500] + contra[0:7500]
    train_nli_labels = ["neutral"]*7500 + ["entailment"]*7500 + ["contradiction"]*7500

    dev_nli_texts = neus[7500:8500] + entails[7500:8500] + contra[7500:8500]
    dev_nli_labels = ["neutral"]*1000 + ["entailment"]*1000 + ["contradiction"]*1000

    file = open("./data/snli_promptonly/train-ur.tsv", "w+")
    for i in range(22500):
        file.write(train_nli_texts[i] + "\t" + train_nli_labels[i] + "\n")
    file.close()

    file = open("./data/snli_promptonly/dev-ur.tsv", "w+")
    for i in range(3000):
        file.write(dev_nli_texts[i] + "\t" + dev_nli_labels[i] + "\n")
    file.close()
    
    #-------------------------------------------------------------------------------------------
    file = open("./data/snli/pretraining/pretrain_label_swprehypos.csv.labeledpretrain.train")
    nli_lines = file.readlines()
    file.close()

    neus, contra, entails = [], [], []
    for line in nli_lines:
        text = line.split("\t")[0].strip() + "\t" + line.split("\t")[1].strip()
        label = line.split("\t")[2].strip().replace("\n", "")
        if label == "entailment": entails.append(text)
        elif label == "neutral": neus.append(text)
        elif label == "contradiction": contra.append(text)

    print(len(neus), len(contra), len(entails))

    assert len(neus) + len(contra) + len(entails) == len(nli_lines)

    random.seed(42)
    random.shuffle(neus)
    random.shuffle(entails)
    random.shuffle(contra)

    train_nli_texts = neus[0:7500] + entails[0:7500] + contra[0:7500]
    train_nli_labels = ["neutral"]*7500 + ["entailment"]*7500 + ["contradiction"]*7500

    dev_nli_texts = neus[7500:8500] + entails[7500:8500] + contra[7500:8500]
    dev_nli_labels = ["neutral"]*1000 + ["entailment"]*1000 + ["contradiction"]*1000

    file = open("./data/snli_promptonly/train-sw.tsv", "w+")
    for i in range(22500):
        file.write(train_nli_texts[i] + "\t" + train_nli_labels[i] + "\n")
    file.close()

    file = open("./data/snli_promptonly/dev-sw.tsv", "w+")
    for i in range(3000):
        file.write(dev_nli_texts[i] + "\t" + dev_nli_labels[i] + "\n")
    file.close()


def make_marsentimentdata():
    file = open("./data/marsentiment/pretraining/pretrain_label_en.csv.labeledpretrain.train")
    lines = file.readlines()
    file.close()

    neutrals, negatives, positives = [], [], []
    for line in lines:
        text = line.split("\t")[0].strip()
        label = line.split("\t")[1].strip().replace("\n", "")
        if label == "positive": positives.append(text)
        elif label == "neutral": neutrals.append(text)
        elif label == "negative": negatives.append(text)

    print(len(neutrals), len(negatives), len(positives))

    assert len(neutrals) + len(negatives) + len(positives) == len(lines)

    random.seed(42)
    random.shuffle(neutrals)
    random.shuffle(positives)
    random.shuffle(negatives)

    train_texts = neutrals[0:5000] + positives[0:5000] + negatives[0:5000]
    train_labels = ["neutral"]*5000 + ["positive"]*5000 + ["negative"]*5000

    dev_texts = neutrals[5000:6000] + positives[5000:6000] + negatives[5000:6000]
    dev_labels = ["neutral"]*1000 + ["positive"]*1000 + ["negative"]*1000

    file = open("./data/marsentiment_promptonly/train-en.tsv", "w+")
    for i in range(15000):
        file.write(train_texts[i] + "\t" + train_labels[i] + "\n")
    file.close()

    file = open("./data/marsentiment_promptonly/dev-en.tsv", "w+")
    for i in range(3000):
        file.write(dev_texts[i] + "\t" + dev_labels[i] + "\n")
    file.close()

    #------------------------------------------------------------------------------------------------------------------

    file = open("./data/marsentiment/pretraining/pretrain_label_mar.csv.labeledpretrain.train")
    lines = file.readlines()
    file.close()

    neutrals, negatives, positives = [], [], []
    for line in lines:
        text = line.split("\t")[0].strip()
        label = line.split("\t")[1].strip().replace("\n", "")
        if label == "positive": positives.append(text)
        elif label == "neutral": neutrals.append(text)
        elif label == "negative": negatives.append(text)

    print(len(neutrals), len(negatives), len(positives))

    assert len(neutrals) + len(negatives) + len(positives) == len(lines)

    random.seed(42)
    random.shuffle(neutrals)
    random.shuffle(positives)
    random.shuffle(negatives)

    train_texts = neutrals[0:5000] + positives[0:5000] + negatives[0:5000]
    train_labels = ["neutral"]*5000 + ["positive"]*5000 + ["negative"]*5000

    dev_texts = neutrals[5000:6000] + positives[5000:6000] + negatives[5000:6000]
    dev_labels = ["neutral"]*1000 + ["positive"]*1000 + ["negative"]*1000

    file = open("./data/marsentiment_promptonly/train-mar.tsv", "w+")
    for i in range(15000):
        file.write(train_texts[i] + "\t" + train_labels[i] + "\n")
    file.close()

    file = open("./data/marsentiment_promptonly/dev-mar.tsv", "w+")
    for i in range(3000):
        file.write(dev_texts[i] + "\t" + dev_labels[i] + "\n")
    file.close()


def make_bnsentimentdata():
    file = open("./data/bnsentiment/pretraining/pretrain_label_en.csv.labeledpretrain.train")
    lines = file.readlines()
    file.close()

    neutrals, negatives, positives = [], [], []
    for line in lines:
        text = line.split("\t")[0].strip()
        label = line.split("\t")[1].strip().replace("\n", "")
        if label == "positive": positives.append(text)
        elif label == "neutral": neutrals.append(text)
        elif label == "negative": negatives.append(text)

    print(len(neutrals), len(negatives), len(positives))

    assert len(neutrals) + len(negatives) + len(positives) == len(lines)

    random.seed(42)
    random.shuffle(neutrals)
    random.shuffle(positives)
    random.shuffle(negatives)

    train_texts = neutrals[0:5000] + positives[0:5000] + negatives[0:5000]
    train_labels = ["neutral"]*5000 + ["positive"]*5000 + ["negative"]*5000

    dev_texts = neutrals[5000:6000] + positives[5000:6000] + negatives[5000:6000]
    dev_labels = ["neutral"]*1000 + ["positive"]*1000 + ["negative"]*1000

    file = open("./data/bnsentiment_promptonly/train-en.tsv", "w+")
    for i in range(15000):
        file.write(train_texts[i] + "\t" + train_labels[i] + "\n")
    file.close()

    file = open("./data/bnsentiment_promptonly/dev-en.tsv", "w+")
    for i in range(3000):
        file.write(dev_texts[i] + "\t" + dev_labels[i] + "\n")
    file.close()

    #------------------------------------------------------------------------------------------------------------------

    file = open("./data/bnsentiment/pretraining/pretrain_label_bn.csv.labeledpretrain.train")
    lines = file.readlines()
    file.close()

    neutrals, negatives, positives = [], [], []
    for line in lines:
        text = line.split("\t")[0].strip()
        label = line.split("\t")[1].strip().replace("\n", "")
        if label == "positive": positives.append(text)
        elif label == "neutral": neutrals.append(text)
        elif label == "negative": negatives.append(text)

    print(len(neutrals), len(negatives), len(positives))

    assert len(neutrals) + len(negatives) + len(positives) == len(lines)

    random.seed(42)
    random.shuffle(neutrals)
    random.shuffle(positives)
    random.shuffle(negatives)

    train_texts = neutrals[0:5000] + positives[0:5000] + negatives[0:5000]
    train_labels = ["neutral"]*5000 + ["positive"]*5000 + ["negative"]*5000

    dev_texts = neutrals[5000:6000] + positives[5000:6000] + negatives[5000:6000]
    dev_labels = ["neutral"]*1000 + ["positive"]*1000 + ["negative"]*1000

    file = open("./data/bnsentiment_promptonly/train-bn.tsv", "w+")
    for i in range(15000):
        file.write(train_texts[i] + "\t" + train_labels[i] + "\n")
    file.close()

    file = open("./data/bnsentiment_promptonly/dev-bn.tsv", "w+")
    for i in range(3000):
        file.write(dev_texts[i] + "\t" + dev_labels[i] + "\n")
    file.close()


def make_hiproductdata():
    file = open("./data/hiproduct/pretraining/pretrain_label_en.csv.labeledpretrain.train.wneutral")
    lines = file.readlines()
    file.close()

    neutrals, negatives, positives = [], [], []
    for line in lines:
        text = line.split("\t")[0].strip()
        label = line.split("\t")[1].strip().replace("\n", "")
        if label == "positive": positives.append(text)
        elif label == "neutral": neutrals.append(text)
        elif label == "negative": negatives.append(text)

    print(len(neutrals), len(negatives), len(positives))

    assert len(neutrals) + len(negatives) + len(positives) == len(lines)

    random.seed(42)
    random.shuffle(neutrals)
    random.shuffle(positives)
    random.shuffle(negatives)

    train_texts = neutrals[0:5000] + positives[0:5000] + negatives[0:5000]
    train_labels = ["neutral"]*5000 + ["positive"]*5000 + ["negative"]*5000

    dev_texts = neutrals[5000:6000] + positives[5000:6000] + negatives[5000:6000]
    dev_labels = ["neutral"]*1000 + ["positive"]*1000 + ["negative"]*1000

    file = open("./data/hiproduct_promptonly/train-en.tsv", "w+")
    for i in range(15000):
        file.write(train_texts[i] + "\t" + train_labels[i] + "\n")
    file.close()

    file = open("./data/hiproduct_promptonly/dev-en.tsv", "w+")
    for i in range(3000):
        file.write(dev_texts[i] + "\t" + dev_labels[i] + "\n")
    file.close()

    #------------------------------------------------------------------------------------------------------------------

    file = open("./data/hiproduct/pretraining/pretrain_label_hi.csv.labeledpretrain.train.wneutral")
    lines = file.readlines()
    file.close()

    neutrals, negatives, positives = [], [], []
    for line in lines:
        text = line.split("\t")[0].strip()
        label = line.split("\t")[1].strip().replace("\n", "")
        if label == "positive": positives.append(text)
        elif label == "neutral": neutrals.append(text)
        elif label == "negative": negatives.append(text)

    print(len(neutrals), len(negatives), len(positives))

    assert len(neutrals) + len(negatives) + len(positives) == len(lines)

    random.seed(42)
    random.shuffle(neutrals)
    random.shuffle(positives)
    random.shuffle(negatives)

    train_texts = neutrals[0:5000] + positives[0:5000] + negatives[0:5000]
    train_labels = ["neutral"]*5000 + ["positive"]*5000 + ["negative"]*5000

    dev_texts = neutrals[5000:6000] + positives[5000:6000] + negatives[5000:6000]
    dev_labels = ["neutral"]*1000 + ["positive"]*1000 + ["negative"]*1000

    file = open("./data/hiproduct_promptonly/train-hi.tsv", "w+")
    for i in range(15000):
        file.write(train_texts[i] + "\t" + train_labels[i] + "\n")
    file.close()

    file = open("./data/hiproduct_promptonly/dev-hi.tsv", "w+")
    for i in range(3000):
        file.write(dev_texts[i] + "\t" + dev_labels[i] + "\n")
    file.close()


def make_gluecosdata():
    file = open("./data/gluecos/pretraining/pretrain_label_en.csv.labeledpretrain.train")
    lines = file.readlines()
    file.close()

    neutrals, negatives, positives = [], [], []
    for line in lines:
        text = line.split("\t")[0].strip()
        label = line.split("\t")[1].strip().replace("\n", "")
        if label == "positive": positives.append(text)
        elif label == "neutral": neutrals.append(text)
        elif label == "negative": negatives.append(text)

    print(len(neutrals), len(negatives), len(positives))

    assert len(neutrals) + len(negatives) + len(positives) == len(lines)

    random.seed(42)
    random.shuffle(neutrals)
    random.shuffle(positives)
    random.shuffle(negatives)

    train_texts = neutrals[0:5000] + positives[0:5000] + negatives[0:5000]
    train_labels = ["neutral"]*5000 + ["positive"]*5000 + ["negative"]*5000

    dev_texts = neutrals[5000:6000] + positives[5000:6000] + negatives[5000:6000]
    dev_labels = ["neutral"]*1000 + ["positive"]*1000 + ["negative"]*1000

    file = open("./data/gluecos_promptonly/train-en.tsv", "w+")
    for i in range(15000):
        file.write(train_texts[i] + "\t" + train_labels[i] + "\n")
    file.close()

    file = open("./data/gluecos_promptonly/dev-en.tsv", "w+")
    for i in range(3000):
        file.write(dev_texts[i] + "\t" + dev_labels[i] + "\n")
    file.close()

    #------------------------------------------------------------------------------------------------------------------

    file = open("./data/gluecos/pretraining/pretrain_label_enhid.csv.labeledpretrain.train")
    lines = file.readlines()
    file.close()

    neutrals, negatives, positives = [], [], []
    for line in lines:
        text = line.split("\t")[0].strip()
        label = line.split("\t")[1].strip().replace("\n", "")
        if label == "positive": positives.append(text)
        elif label == "neutral": neutrals.append(text)
        elif label == "negative": negatives.append(text)

    print(len(neutrals), len(negatives), len(positives))

    assert len(neutrals) + len(negatives) + len(positives) == len(lines)

    random.seed(42)
    random.shuffle(neutrals)
    random.shuffle(positives)
    random.shuffle(negatives)

    train_texts = neutrals[0:5000] + positives[0:5000] + negatives[0:5000]
    train_labels = ["neutral"]*5000 + ["positive"]*5000 + ["negative"]*5000

    dev_texts = neutrals[5000:6000] + positives[5000:6000] + negatives[5000:6000]
    dev_labels = ["neutral"]*1000 + ["positive"]*1000 + ["negative"]*1000

    file = open("./data/gluecos_promptonly/train-hid.tsv", "w+")
    for i in range(15000):
        file.write(train_texts[i] + "\t" + train_labels[i] + "\n")
    file.close()

    file = open("./data/gluecos_promptonly/dev-hid.tsv", "w+")
    for i in range(3000):
        file.write(dev_texts[i] + "\t" + dev_labels[i] + "\n")
    file.close()

make_snlidata()
# make_marsentimentdata()
# make_bnsentimentdata()
# make_hiproductdata()
# make_gluecosdata()