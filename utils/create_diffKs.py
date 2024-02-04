file = open("./data/marsentiment/zero-shot_topk/expt-2/5k/train-mar.tsv")
lines = file.readlines()
file.close()

neutrals = lines[0:5000]
positives = lines[5000:10000]
negatives = lines[10000:15000]

train_lines = lines[15000:]

file = open("./data/marsentiment/zero-shot_topk/expt-2/2k/train-mar.tsv", "w+")
for sent in neutrals[0:2000]:
    file.write(sent)
for sent in positives[0:2000]:
    file.write(sent)
for sent in negatives[0:2000]:
    file.write(sent)
for sent in train_lines:
    file.write(sent)
file.close()

file = open("./data/marsentiment/zero-shot_topk/expt-2/3k/train-mar.tsv", "w+")
for sent in neutrals[0:3000]:
    file.write(sent)
for sent in positives[0:3000]:
    file.write(sent)
for sent in negatives[0:3000]:
    file.write(sent)
for sent in train_lines:
    file.write(sent)
file.close()

file = open("./data/marsentiment/zero-shot_topk/expt-2/3.5k/train-mar.tsv", "w+")
for sent in neutrals[0:3500]:
    file.write(sent)
for sent in positives[0:3500]:
    file.write(sent)
for sent in negatives[0:3500]:
    file.write(sent)
for sent in train_lines:
    file.write(sent)
file.close()

file = open("./data/marsentiment/zero-shot_topk/expt-2/4k/train-mar.tsv", "w+")
for sent in neutrals[0:4000]:
    file.write(sent)
for sent in positives[0:4000]:
    file.write(sent)
for sent in negatives[0:4000]:
    file.write(sent)
for sent in train_lines:
    file.write(sent)
file.close()

file = open("./data/marsentiment/zero-shot_topk/expt-2/4.5k/train-mar.tsv", "w+")
for sent in neutrals[0:4500]:
    file.write(sent)
for sent in positives[0:4500]:
    file.write(sent)
for sent in negatives[0:4500]:
    file.write(sent)
for sent in train_lines:
    file.write(sent)
file.close()