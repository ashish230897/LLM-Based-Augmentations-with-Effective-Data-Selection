file = open("/raid/speech/ashish/TSTG_new/data/snli/pretraining/zero-shot_topk_2.5k_each_hi_hisnlift.pl")
lines = file.readlines()
file.close()

lines = [line.split("\t")[0] + "\t" + line.split("\t")[1] for line in lines]

file = open("/raid/speech/ashish/TSTG_new/data/snli/zero-shot_topk/expt-2/topk_2.5k_each.hi", "w+")
for line in lines:
    file.write(line.strip().replace("\n", "") + "\n")
file.close()