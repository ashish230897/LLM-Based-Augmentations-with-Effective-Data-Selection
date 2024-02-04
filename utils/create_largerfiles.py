import random

type = "rand"
expt = 2
lang = "mar"

file = open("./data/marsentiment/diverseK/pretrain_label_{}.csv.pretrain.train.pl".format(lang))
lines = file.readlines()
file.close()

pos_pairs, neu_pairs, neg_pairs = [], [], []
for line in lines:
    text = line.split("\t")[0].strip().replace("\n", "")
    label = line.split("\t")[1].strip().replace("\n", "")
    prob = float(line.split("\t")[2].strip().replace("\n", ""))
    
    if label == "positive": pos_pairs.append([text, prob])
    elif label == "negative": neg_pairs.append([text, prob])
    elif label == "neutral": neu_pairs.append([text, prob])
    else: print("not found!!!")

print(len(neu_pairs))

if type == "rand":
    random.seed(42)

    random.shuffle(neu_pairs)
    random.shuffle(pos_pairs)
    random.shuffle(neg_pairs)
    
    file = open("./data/marsentiment/zero-shot_{}k/expt-{}/{}k_5k_each.{}".format(type, expt, type, lang), "w+")
    for pair in neu_pairs[0:5000]:
        file.write(pair[0] + "\n")
    for pair in pos_pairs[0:5000]:
        file.write(pair[0] + "\n")
    for pair in neg_pairs[0:5000]:
        file.write(pair[0] + "\n")
    file.close()
    
    file = open("./data/marsentiment/zero-shot_{}k/expt-{}/{}k_7.5k_each.{}".format(type, expt, type, lang), "w+")
    for pair in neu_pairs[0:7500]:
        file.write(pair[0] + "\n")
    for pair in pos_pairs[0:7500]:
        file.write(pair[0] + "\n")
    for pair in neg_pairs[0:7500]:
        file.write(pair[0] + "\n")
    file.close()
    
    
    
    
    
# print(pos_pairs[0])
# sorted(pos_pairs, lambda x: x[1], reverse=True)
# print(pos_pairs[0])
    
# print(neg_pairs[0])
# sorted(neg_pairs, lambda x: x[1], reverse=True)
# print(neg_pairs[0])

# print(neu_pairs[0])
# sorted(neu_pairs, lambda x: x[1], reverse=True)
# print(neu_pairs[0])

