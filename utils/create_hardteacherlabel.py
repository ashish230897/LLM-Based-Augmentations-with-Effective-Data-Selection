file = open("/raid/speech/ashish/TSTG_new/data/marsentiment/zero-shot_rand/expt-2/train-mar.tsv")
lines = file.readlines()
file.close()

file = open("/raid/speech/ashish/TSTG_new/data/marsentiment/pretraining/zero-shot_rand_2.5k_each_mar_marsst5ft.pl")
augment_lines = file.readlines()
file.close()

augment_dict = {}  # mapping text to labels from teacher selections
for line in augment_lines:
    augment_dict[line.split("\t")[0].strip().replace("\n", "")] = line.split("\t")[1].strip().replace("\n", "")


positive, negative, neutral = 0, 0, 0

file = open("/raid/speech/ashish/TSTG_new/data/marsentiment/zero-shot_rand/expt-2/hard_teacher/train-mar.tsv", "w+")
final_lines = []
for i,line in enumerate(lines):
    text, label, type = line.split("\t")[0].strip().replace("\n", ""), line.split("\t")[1].strip().replace("\n", ""), line.split("\t")[2].strip().replace("\n", "") 

    if str(type) == "0":
        if text not in augment_dict: print("alert!")
        new_line = text + "\t" + augment_dict[text] + "\t" + "1\n"
        # new_line = text + "\t" + augment_dict[text] + "\n"
        if augment_dict[text] == "positive": positive += 1
        elif augment_dict[text] == "neutral": neutral += 1
        elif augment_dict[text] == "negative": negative += 1
        final_lines.append(new_line)
    else:
        final_lines.append(text + "\t" + label + "\t" + "1\n")

for line in final_lines:
    file.write(line)
file.close()

print(positive, negative, neutral)