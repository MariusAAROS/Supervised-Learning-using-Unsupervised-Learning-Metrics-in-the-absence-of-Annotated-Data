import pandas as pd

def cleanString(string, maxSpacing=10):
    """
    Remove noisy and useless characters from a given string.

    :param1 string (string): Initial corpus
    :param2 maxSpacing (int): Maximal number of adjacent space to be found and suppressed in the corpus.

    :output clean (string): Cleansed corpus
    """

    clean = string[:]

    #remove linebreaks and superfluous characters 
    clean = clean.replace("\n", " ")
    clean = clean.replace("-", "")
    clean = clean.replace("''", "")
    clean = clean.replace("``", "")
    clean = clean.replace('""', "")
    clean = clean.replace("...", ".")
    clean = clean.replace("..", ".")
    clean = clean.replace("_", "")
    clean = clean.replace("<S>", "")
    clean = clean.replace(" .", ".")

    #remove overspacing
    spacing = "".join([" " for _ in range(maxSpacing)])
    for _ in range(maxSpacing-1):
        spacing = spacing[:-1]
        clean = clean.replace(spacing, " ")
    
    return clean

def cleanDataset(dataset):
    """
    Cleanses a pandas DataFrame.

    :param1 dataset (DataFrame): pandas DataFrame containing at least a <text> and a <summary> column.
    """
    dataset.loc[:,"text"] = dataset["summary"].replace(regex=r"\[[^\]]*\]", value="")
    dataset.loc[:,"text"] = dataset["summary"].replace(regex=r"\[[^\]]*\]", value="")
    dataset.loc[:,"text"] = dataset["summary"].map(cleanString)
    dataset.loc[:,"text"] = dataset["summary"].map(cleanString)
   