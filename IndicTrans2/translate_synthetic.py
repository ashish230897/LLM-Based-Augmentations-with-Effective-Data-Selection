from inference.engine import Model
import pandas as pd
import time
from tqdm import tqdm

repo_path = "/raid/speech/ashish/TSTG_new" 
ckpt_dir = "{}/translations/en-indic-preprint/fairseq_model/".format(repo_path)

model = Model(ckpt_dir, model_type="fairseq")


def translate_synthetic_data(lang):
    data_path = "{}/data/xnli/pretraining/processed.csv".format(repo_path)
    premises, hypos = [], []

    df = pd.read_csv(data_path)
    premises, hypos = list(df["Premises"]), list(df["Hypothesis"])
    
    print(len(premises), len(hypos))

    lang_code_dict = {"hi":"hin_Deva", "en":"eng_Latn", "ur": "urd_Arab", "mr":"mar_Deva"}
    batch_size = 400
    num_batches = int(len(premises)/batch_size)
    print("Number of batches are {}".format(num_batches))

    
    t1 = time.time()
    
    # translate premises
    premises_txns = []
    for i in tqdm(range(num_batches)):
        if i == num_batches-1:
            curr_batch = premises[i*batch_size:]
        else:
            curr_batch = premises[i*batch_size:i*batch_size+batch_size]

        try:            
            curr_translations = model.batch_translate(curr_batch, "eng_Latn", lang_code_dict[lang])
        except Exception as e:
            print("Error is ", e)
        
        
        premises_txns += curr_translations
    
    assert len(premises) == len(premises_txns)


    # translate hypothesis
    hypos_txns = []
    for i in tqdm(range(num_batches)):
        if i == num_batches-1:
            curr_batch = hypos[i*batch_size:]
        else:
            curr_batch = hypos[i*batch_size:i*batch_size+batch_size]

        try:
            curr_translations = model.batch_translate(curr_batch, "eng_Latn", lang_code_dict[lang])
        except Exception as e:
            print("Error is ", e)
        
        hypos_txns += curr_translations
    
    assert len(hypos) == len(hypos_txns)

    dict = {"Premises": premises_txns, "Hypothesis": hypos_txns}
    df = pd.DataFrame(dict)
    df.to_csv("{}/data/xnli/pretraining/processed_{}.csv".format(repo_path, lang), index=False)

    print("Time taken for lang {} is {}".format(lang, time.time() - t1))

def main():
    translate_synthetic_data("mr")


if __name__ == "__main__":
    main()