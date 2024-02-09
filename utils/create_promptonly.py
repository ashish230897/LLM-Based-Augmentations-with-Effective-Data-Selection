import random

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
random.shuffle(contra)
random.shuffle(entails)

train_nli_texts = neus[0:2500] + entails[0:2500] + contra[0:2500]
train_nli_labels = ["neutral"]*2500 + ["entailment"]*2500 + ["contradiction"]*2500

dev_nli_texts = neus[2500:3000] + entails[2500:3000] + contra[2500:3000]
dev_nli_labels = ["neutral"]*500 + ["entailment"]*500 + ["contradiction"]*500

file = open("./data/snli_promptonly/train-en.tsv", "w+")
for i in range(7500):
    file.write(train_nli_texts[i] + "\t" + train_nli_labels[i] + "\n")
file.close()

file = open("./data/snli_promptonly/dev-en.tsv", "w+")
for i in range(1500):
    file.write(dev_nli_texts[i] + "\t" + dev_nli_labels[i] + "\n")
file.close()


############## now saving marsentiment task
file = open("./data/marsentiment/pretraining/pretrain_label_en.csv.labeledpretrain.train")
lines = file.readlines()
file.close()

neus, negs, pos = [], [], []
for line in lines:
    text = line.split("\t")[0].strip()
    label = line.split("\t")[1].strip().replace("\n", "")
    if label == "positive": pos.append(text)
    elif label == "neutral": neus.append(text)
    elif label == "negative": negs.append(text)

print(len(neus), len(negs), len(pos))

assert len(neus) + len(negs) + len(pos) == len(lines)

random.seed(42)
random.shuffle(neus)
random.shuffle(negs)
random.shuffle(pos)

train_sentiment_texts = neus[0:2500] + pos[0:2500] + negs[0:2500]
train_sentiment_labels = ["neutral"]*2500 + ["positive"]*2500 + ["negative"]*2500

dev_sentiment_texts = neus[2500:3000] + pos[2500:3000] + negs[2500:3000]
dev_sentiment_labels = ["neutral"]*500 + ["positive"]*500 + ["negative"]*500

file = open("./data/marsentiment_promptonly/train-en.tsv", "w+")
for i in range(7500):
    file.write(train_sentiment_texts[i] + "\t" + train_sentiment_labels[i] + "\n")
file.close()

file = open("./data/marsentiment_promptonly/dev-en.tsv", "w+")
for i in range(1500):
    file.write(dev_sentiment_texts[i] + "\t" + dev_sentiment_labels[i] + "\n")
file.close()