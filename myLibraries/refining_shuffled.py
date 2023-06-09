from custom_score.refine import Refiner
import pandas as pd
import numpy as np
from custom_score.utils import model_load
from custom_score.score import score
import bert_score

#params
size = 500
dataset_name = "Billsum"
save = True
savePace = 50

#url dictionnary
datasets_list = {"Billsum": 'https://drive.google.com/file/d/1Wd0M3qepNF6B4YwFYrpo7CaSERpudAG_/view?usp=share_link', 
                 "Pubmed": r'D:\COURS\A4\S8 - ESILV\Stage\Work\Datasets\Summary Evaluation\Pubmed\test.json'}

#load dataset
if dataset_name == "Billsum":
    dataset_url=datasets_list[dataset_name]
    dataset_url='https://drive.google.com/uc?export=download&id=' + dataset_url.split('/')[-2]
    dataset = pd.read_json(dataset_url, lines=True)
    dataset = dataset.loc[:, ["text", "summary"]]

elif dataset_name == "Pubmed":
    dataset_url=datasets_list[dataset_name]
    dataset = pd.read_json(dataset_url, lines=True)
    dataset = dataset[["article_text", "abstract_text"]]
    cleaner = lambda x: ". ".join(x).replace("<S>", "").strip()
    dataset.loc[:,"abstract_text"] = dataset["abstract_text"].replace(regex=r"\[[^\]]*\]", value="")
    dataset.loc[:,"article_text"] = dataset["article_text"].replace(regex=r"\[[^\]]*\]", value="")
    dataset.loc[:,"abstract_text"] = dataset["abstract_text"].map(cleaner)
    dataset.loc[:,"article_text"] = dataset["article_text"].map(cleaner)
    dataset = dataset.rename(columns={"abstract_text": "summary",
                            "article_text": "text"})

dataset["summary"] = dataset["summary"].sample(frac=1).values
subset = dataset.iloc[:size, :]

#refine
w2v = model_load("Word2Vec", True)
r = Refiner(corpus=subset["text"].to_list(), gold=subset["summary"].to_list(), 
            model=w2v, metric=bert_score.score, dist_metric=score, ratio=3, 
            maxSpacing=15, printRange=range(0, 3)) #ratio=np.linspace(2, 3, 2)
r.refine(checkpoints=save, saveRate=savePace)