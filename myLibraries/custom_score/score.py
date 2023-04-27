from .utils import *
import numpy as np
from datetime import datetime
import bert_score 

def score(model, candidates=["I am Marius"], references=["Marius is my name"], withIdf = False):
    """
    Computes BERTScore using a custom embedding amongst Word2Vec, Fasttext and Glove.

    :param1 model (dict): Dictionnary of the embedding.
    :param2 references (list): List of reference sentences.
    :param3 candidates (list): List of candidate sentences.

    :output formatedScores (List): List containing tuples of R, P and F for each couple of the current corpus.
    """
    #storing raw references for IDF Calculus
    raw_references = [reference for reference in references]

    #parsing non-encoded sentences
    word_references = []
    word_candidates = []
    for reference, candidate in zip(references, candidates):
        word_references.append(reference.split(" "))
        word_candidates.append(candidate.split(" "))

    #encoding to vectors
    references, _ = encode(references, model)
    candidates, _ = encode(candidates, model)

    #cosine similarity
    candToRef = similarityCandToRef(references, candidates)

    refToCand = []
    for similarityMatrix in candToRef:
        refToCand.append(np.transpose(similarityMatrix))

    allIdfDicts = []
    for reference in raw_references:
        allIdfDicts.append(computeIdf(reference))

    #Metrics calculation
    if withIdf == True:
        (R, P, F) = computeMetricsWithIdf(refToCand, candToRef, word_references, word_candidates, allIdfDicts)
    else:
        (R, P, F) = computeMetrics(refToCand, candToRef, references, candidates)

    formatedScores = [(r, p, f) for r, p, f in zip(R, P, F)]
    return formatedScores

def BERTScoreDynamicSampleTest(data, limit=3, modelPath = None, model = None, nbLayers = 24):
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

def BERTScoreStaticSampleTest(data, model, limit=3, withIdf = False):
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
        
        out = score(model, curCand, curRef, withIdf=withIdf)
        scores.append(out[0])
        if nbIter >= limit:
            break
        nbIter += 1
    runtime = (datetime.now() - init_time).total_seconds()
    return scores, runtime