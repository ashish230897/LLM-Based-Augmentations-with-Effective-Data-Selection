import pandas as pd

def convert_text(df):
    texts = []
    
    for _, row in df.iterrows():
        premise, hypo = row["premise"], row["hypothesis"]
        text = "Premise: {} \nHypothesis: {}".format(premise, hypo)
        texts.append(text)
    
    df["text"] = texts
    
    return df


def main():
    df = pd.read_csv("./data/test-en.tsv", delimiter="\t", names=["premise", "hypothesis", "label"])
    df = convert_text(df)

    print("First row is:")
    print(df.iloc[0])

    df.to_csv("./data/test-en-texts.csv", index=False)

if __name__ == "__main__":
    main()

