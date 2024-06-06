#file = open("./data/snli/zero-shot_topk/expt-2/hard/train-sw.tsv")
file = open("./data/snli/diverseK/hard/train-en.tsv")
lines = file.readlines()
file.close()

teacher_labels = ["neutral"]*2500 + ["entailment"]*2500 + ["contradiction"]*2500
prompt_labels = []
matched_texts, matched_labels = [], []

for i in range(7500):
    label = lines[i].split("\t")[2]
    text = lines[i].split("\t")[0].strip() + "\t" + lines[i].split("\t")[1].strip()

    if label == teacher_labels[i]:
        matched_texts.append(text)
        matched_labels.append(label)


#file = open("./data/snli/zero-shot_topk/expt-2/overlap_hard/train-sw.tsv", "w+")
file = open("./data/snli/diverseK/overlap_hard/train-en.tsv", "w+")
for i in range(len(matched_labels)):
    file.write(matched_texts[i] + "\t" + matched_labels[i] + "\t" + "1\n")

for i in range(7500, len(lines)):
    file.write(lines[i])
file.close()

#file = open("./data/snli/zero-shot_topk/expt-2/overlap_soft/train-sw.tsv", "w+")
file = open("./data/snli/diverseK/overlap_soft/train-en.tsv", "w+")
for i in range(len(matched_labels)):
    file.write(matched_texts[i] + "\t" + matched_labels[i] + "\t" + "0\n")

for i in range(7500, len(lines)):
    file.write(lines[i])
file.close()