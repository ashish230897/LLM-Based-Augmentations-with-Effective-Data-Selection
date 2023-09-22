import pandas as pd
import argparse

def main(input_path, output_path):

    df = pd.read_csv(input_path)
    texts = list(df["Texts"])

    texts = set(texts)
    texts = [text for text in texts if len(text) > 2]
    
    dict = {"Texts": texts}

    df_new = pd.DataFrame(dict)
    df_new.to_csv(output_path, index=False)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    args = parser.parse_args()

    main(args.input_path, args.output_path)
