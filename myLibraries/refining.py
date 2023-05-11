from custom_score.refine import Refiner
import pandas as pd
import numpy as np
from custom_score.utils import serialized_to_model
from custom_score.score import score
import os
from datetime import datetime


#params
size = 2

#utils
def updateFileCount(path):
    """
    :param1 path (string): Path to the file counting the number of experimentation output I already generated.

    :output updated (int): New number of output file created after the current experimentation.
    """
    if os.path.exists(path):
        with open(path, "r") as f:
            value = int(f.read())
        updated = value + 1
    else:
        updated = 1
    with open(path, "w") as f:
        f.write(str(updated))    
    return updated

#load dataset
billsum_path = r"C:\Pro\Stages\A4 - DVRC\Work\Datasets\billsum"
billsum_test = pd.read_json(path_or_buf = billsum_path + r"\us_test_data_final_OFFICIAL.jsonl", lines=True)
billsum_test = billsum_test.loc[:, ["text", "summary"]]
subset = billsum_test.iloc[:size, :]

#refine
start = datetime.now()
w2v = serialized_to_model(r'C:\Pro\Stages\A4 - DVRC\Work\Models\serialized_w2v.pkl')
r = Refiner(subset["text"], w2v, score, ratio=np.linspace(1, 3, 2), maxSpacing=15, printRange=range(0, 3))
r.refine()
assessement = r.assess()
stop = datetime.now()

#runtime
runtime = (stop - start)

#write output
main_folder_path = r"C:\Pro\Stages\A4 - DVRC\Work\Supervised-Learning-using-Unsupervised-Learning-Metrics-in-the-absence-of-Annotated-Data\myLibraries\refining_output"
countfile_name = r"count.txt"
count = updateFileCount(os.path.join(main_folder_path, countfile_name))

current_path = os.path.join(main_folder_path, f"experimentation_{count}")
os.mkdir(current_path)

#mainDf = r.to_dataframe()
scoreDf = assessement["scores"]
corDf = assessement["correlations"]

#mainDf.to_csv(os.path.join(current_path, "main.csv"))
scoreDf.to_csv(os.path.join(current_path, "scores.csv"))
corDf.to_csv(os.path.join(current_path, "correlations.csv"))
with open(os.path.join(current_path, "runtimes.txt"), "w") as f:
    f.write(str(runtime))