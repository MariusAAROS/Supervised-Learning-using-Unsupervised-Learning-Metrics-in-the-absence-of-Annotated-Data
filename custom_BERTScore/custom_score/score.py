from .utils import *
import numpy as np

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
    candToRef = SimilarityCandToRef(references, candidates)

    refToCand = []
    for similarityMatrix in candToRef:
        refToCand.append(np.transpose(similarityMatrix))
        
    #Metrics calculation
    (R, P, F) = computeMetrics(refToCand, candToRef, references, candidates)

    return (R, P, F)