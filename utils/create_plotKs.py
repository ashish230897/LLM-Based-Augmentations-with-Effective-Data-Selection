import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def matplotlib_lineplot():

    # Generate sample data
    x_values = ["2.5k", "7.5k", "12.5k", "17.5k", "22.5k"]
    dev_topk_en_values = [66.34, 65.04, 65.3, 66.65]
    test_topk_en_values = [66.54, 65.04, 64.3, 66.11]
    test_randk_en_values = [65.42, 65.12, 65.01, 66.3, 65.33]
    test_hybk_en_values = [66.12, 66.03, 64.26, 65.68, 66.11]

    dev_hybk_en_values = [66.38, 66.56, 65.49, 66.25, 66.46]
    dev_randk_en_values = [65.38, 65.29, 64.97, 65.59, 65.6]

    dev_topk_lang_values = [65.68, 65.39, 64.48, 62.33, 64.79]
    test_topk_lang_values = [66.53, 65.97, 65.06, 63.46, 65.42]
    test_randk_lang_values = [65.91, 65.07, 64.42, 65.2, 65.55]
    test_hybk_lang_values = [65.99, 65.59, 66.12, 66.17, 66.07]

    dev_randk_lang_values = [65.38, 64.31, 63.57, 65.14, 64.08]
    dev_hybk_lang_values = [65.81, 66.32, 65.28, 64.73, 64.64]

    # Create the plot
    plt.figure(figsize=(8, 6))

    # plt.plot(x_values, dev_topk_values, label='dev mar-topk', color='blue', linestyle='-')
    # plt.plot(x_values, dev_topk_lang_values, label='dev lang-topk', color='red', linestyle='--')
    plt.plot(x_values, dev_hybk_lang_values, label='trans-hybk', color='green', linestyle='-.')
    plt.plot(x_values, dev_randk_lang_values, label='trans-promptrandk', color='purple', linestyle=':')

    # Add labels and title
    plt.xlabel('Augmented data size')
    plt.ylabel('Accuracies')
    plt.title('Marathi sentiment test accuracies')

    # Add a legend
    plt.legend()
    plt.savefig("different_ks_trans_marsentiment_dev.png")
    

def sns_lineplot():
    
    mar_sentiment_scores = [0.743, 0.832, 0.826]
    xnli_scores = [0.743, 0.752, 0.758]
    hiproduct_scores = [0.674, 0.726, 0.732]
    
    x = ["top-k", "rand-k", "div-k"]
    
    df = pd.DataFrame({"selection strategy": x*3, "tasks": ["marsentiment"]*3 + ["xnli"]*3 + ["hiproduct"]*3, 
                                     "diversity scores": mar_sentiment_scores + xnli_scores + hiproduct_scores})
    #df = df.pivot(index="selection strategy", columns="tasks", values="diversity scores")
    
    fig, ax = plt.subplots()
    sns.lineplot(df, y="diversity scores", x="selection strategy", hue="tasks")
    #ax.set_xlim([0, 0.55])
    plt.savefig("./utils/diversity.png")
    plt.clf()

sns_lineplot()