from .utils import *
import numpy as np
from datetime import datetime
import bert_score 

def score(model, candidates=["I am Marius"], references=["Marius is my name"]):
    """
    Computes BERTScore using a custom embedding amongst Word2Vec, Fasttext and Glove.

    :param1 model (dict): Dictionnary of the embedding.
    :param2 references (list): List of reference sentences.
    :param3 candidates (list): List of candidate sentences.

    :output (tuple): Tuple containing R, P and F for each couple of the current corpus.
    """

    #encoding to vectors
    references, _ = encode(references, model)
    candidates, _ = encode(candidates, model)

    #cosine similarity
    candToRef = similarityCandToRef(references, candidates)

    refToCand = []
    for similarityMatrix in candToRef:
        refToCand.append(np.transpose(similarityMatrix))
        
    #Metrics calculation
    (R, P, F) = computeMetrics(refToCand, candToRef, references, candidates)

    return (R, P, F)

def DynamicEmbeddingSampleTest(data, limit=3, modelPath = None, model = None, nbLayers = 24):
    """
    Benchmarking function allowing to compute classical bertscore as well as its runtime.

    :param1 data (DataFrame) : Dataframe containing all references and candidates. Required Format : [col0: Reference, col1: Candidate].
    :param2 limit (int): Number of individuals to compute.
    :param3 modelPath (string): Path to the wanted model in the HuggingFace repository.
    :param4 model (object): Model to use directly for computation.
    :param5 nbLayers (int): Number of layers in the custom model (has to be filled-in if modelPath!=None or model!=None).  
    
    :output1 scores (list): List of Precision, Recall and F1score for each computed individual.
    :output2 runtime (float): Elasped time between the start and end of score computation for all individuals.
    """

    nbIter = 1
    scores = []
    init_time = datetime.now()
    for row in data.iterrows():
        curCand = [" ".join(row[1][1].split("\n"))]
        curRef = [" ".join(row[1][0].split("\n"))]
        assert len(curCand) == len(curRef)
        if modelPath != None:
            (P, R, F), hashname = bert_score.score(curCand, curRef, lang="en", 
                                        model_type=modelPath, 
                                        num_layers=nbLayers, return_hash=True)
        elif model != None:
            (P, R, F), hashname = bert_score.score(curCand, curRef, lang="en", 
                                        model_type=model, 
                                        num_layers=nbLayers, return_hash=True)
        else:
            (P, R, F), hashname = bert_score.score(curCand, curRef, lang="en", return_hash=True)
        P = P[0].item()
        R = R[0].item()
        F = F[0].item()
        scores.append((P, R, F))
        if nbIter >= limit:
            break
        nbIter += 1
    runtime = (datetime.now() - init_time).total_seconds()
    return scores, runtime

def StaticEmbeddingSampleTest(data, model, limit=3):
    """
    Benchmarking function allowing to compute static bertscore as well as its runtime.

    :param1 data (DataFrame) : Dataframe containing all references and candidates. 
    :param2 model (dict): Dictionnary of the embedding. 
    :param3 limit (int): Number of individuals to compute.
    
    :output1 scores (list): List of Precision, Recall and F1score for each computed individual.
    :output2 runtime (float): Elasped time between the start and end of score computation for all individuals.
    """

    nbIter = 1
    scores = []
    init_time = datetime.now()
    for row in data.iterrows():
        curCand = [" ".join(row[1][1].split("\n"))]
        curRef = [" ".join(row[1][0].split("\n"))]
        assert len(curCand) == len(curRef)
        
        (P, R, F) = score(model, curCand, curRef)
        P = P[0]
        R = R[0]
        F = F[0]
        scores.append((P, R, F))

        if nbIter >= limit:
            break
        nbIter += 1
    runtime = (datetime.now() - init_time).total_seconds()
    return scores, runtime