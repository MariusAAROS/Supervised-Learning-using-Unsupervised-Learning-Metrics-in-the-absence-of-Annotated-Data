import git 
import os

def get_git_root():
    """
    Return the path to the current git directory's root.

    output git_root (str): Path of the current git directory's root.
    """
    git_repo = git.Repo(os.path.abspath(__file__), search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root

import sys
sys.path.append(os.path.join(get_git_root(), "myLibraries"))
import pandas as pd
from custom_score.utils import cleanString


def load_billsum():
    """
    Loads Billsum Congressional dataset to a pandas DataFrame.

    output dataset (pd.DataFrame): Billsum dataset.  
    """
    dataset_url="https://drive.google.com/file/d/1Wd0M3qepNF6B4YwFYrpo7CaSERpudAG_/view?usp=share_link"
    dataset_url='https://drive.google.com/uc?export=download&id=' + dataset_url.split('/')[-2]
    dataset = pd.read_json(dataset_url, lines=True)
    dataset = dataset.loc[:, ["text", "summary"]]
    dataset["text"] =  [cleanString(text) for text in dataset["text"].to_list()] 
    return dataset