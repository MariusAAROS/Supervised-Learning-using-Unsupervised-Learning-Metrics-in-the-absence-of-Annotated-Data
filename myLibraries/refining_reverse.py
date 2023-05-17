from custom_score.refine import Refiner
import pandas as pd
import numpy as np
from custom_score.utils import model_load, get_git_root
from custom_score.score import score
import os
from datetime import datetime


#params
size = 500
save = True
savePace = 50

#load dataset
billsumTest_url='https://drive.google.com/file/d/1Wd0M3qepNF6B4YwFYrpo7CaSERpudAG_/view?usp=share_link'
billsumTest_url='https://drive.google.com/uc?id=' + billsumTest_url.split('/')[-2]
billsum_test = pd.read_json(billsumTest_url, lines=True)
billsum_test = billsum_test.loc[:, ["text", "summary"]]
billsum_test["summary"] = billsum_test["summary"].sample(frac=1).values
subset = billsum_test.iloc[:size, :]

#refine
#start = datetime.now()
w2v = model_load("Word2Vec", True)
r = Refiner(subset["text"].to_list(), subset["summary"].to_list(), w2v, score, ratio=np.linspace(1, 3, 2), maxSpacing=15, printRange=range(0, 3))
r.refine(checkpoints=save, saveRate=savePace)