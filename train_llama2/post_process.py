import pandas as pd
import os

path = os.getcwd()

def main():
    df = pd.read_csv("/home/ashish/taskspecificgenerations/results/generations_label_sst5chat_7bquantized.csv")
    texts = list(df["Texts"])

    new_texts = []
    for text in texts:
        if '"' in text:
            new_texts.append(text.split('"')[1])
        else:
            text = text.split('[/INST]')[1]
            if ":" in text:
                new_texts.append(text.split(":")[1])
            else:
                new_texts.append(text)
    
    dict = {"Texts": new_texts}
    df_new = pd.DataFrame(dict)
    df_new.to_csv("{}/results/generations_label_sst5chat_7bquantized_processed.csv".format(path), index=False)



if __name__ == "__main__":
    main()