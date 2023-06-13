from MARScore.score import MARSCore
from MARScore.utils import get_git_root
from custom_score.utils import cleanString
import os
import pandas as pd

#params
size = 20
dataset_name = "Pubmed"

#url dictionnary
datasets_list = {"Billsum": 'https://drive.google.com/file/d/1Wd0M3qepNF6B4YwFYrpo7CaSERpudAG_/view?usp=share_link', 
                 "Pubmed": r'C:\Pro\Stages\A4 - DVRC\Work\Datasets\pubmed\test.json'}

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
    format_dot = lambda x: x.replace(" .", ".")
    dataset.loc[:,"abstract_text"] = dataset["abstract_text"].replace(regex=r"\[[^\]]*\]", value="")
    dataset.loc[:,"article_text"] = dataset["article_text"].replace(regex=r"\[[^\]]*\]", value="")
    dataset.loc[:,"abstract_text"] = dataset["abstract_text"].map(cleaner)
    dataset.loc[:,"article_text"] = dataset["article_text"].map(cleaner)
    dataset.loc[:,"abstract_text"] = dataset["abstract_text"].map(cleanString)
    dataset.loc[:,"article_text"] = dataset["article_text"].map(cleanString)
    dataset.loc[:,"abstract_text"] = dataset["abstract_text"].map(format_dot)
    dataset.loc[:,"article_text"] = dataset["article_text"].map(format_dot)
    dataset = dataset.rename(columns={"abstract_text": "summary",
                            "article_text": "text"})
    
subset = dataset.iloc[:size, :]

#refine
ms = MARSCore(subset["text"].to_list(), subset["summary"].to_list())
ms.compute()
res = ms.assess()
scores = res["scores"]
correlations = res["correlations"]
scores.to_csv(os.path.join(get_git_root(), r"\myLibraries\MARScore_output\results\scores.csv"))
correlations.to_csv(os.path.join(get_git_root(), r"\myLibraries\MARScore_output\results\correlations.csv"))