infile = open('/raid/speech/ashish/TSTG_new/data/snli/pretraining/pretrain_label_hiprehypos.csv.pretrain.train','r')
outfile = open('/raid/speech/ashish/TSTG_new/data/snli/pretraining/pretrain_label_hiprehypos.csv.pretrain.train.wl','w')

lines = infile.readlines()
for line in lines:
    new_line = line.strip()+"\tneutral\n"
    outfile.write(new_line)