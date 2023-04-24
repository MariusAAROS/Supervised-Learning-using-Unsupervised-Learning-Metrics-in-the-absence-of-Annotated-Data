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

    :output (tuple): Tuple containing R, P and F for each couple of the current corpus.
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

def StaticEmbeddingSampleTest(data, model, limit=3, withIdf = False):
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
        
        (P, R, F) = score(model, curCand, curRef, withIdf=withIdf)
        P = P[0]
        R = R[0]
        F = F[0]
        scores.append((P, R, F))

        if nbIter >= limit:
            break
        nbIter += 1
    runtime = (datetime.now() - init_time).total_seconds()
    return scores, runtime

def simplify(reference, model, reductionFactor=2, maxSpacing=10):
    """
    Return a reduced string computed using static embedding vectors similarity. Also denoises the data by removing superfluous elements such as "\n" or useless signs.
 
    :param1 reference (string): Document to simplify.
    :param2 model (dict): Dictionnary of keyed-vectors.
    :param3 reductionFactor (float or int): Number determining how much the reference text will be shortened. 
    :param4 maxSpacing (int): Maximal number of adjacent space to be found and suppressed in the corpus.

    :output simplified (string): Simplified version of the initial document.
    """

    #preprocess corpus
    clean = cleanString(reference, maxSpacing)
    sentences = clean.split(".")
    sentences.pop()
    respaced_sentences = []
    for sentence in sentences:
        if sentence[0] == " ":
            sentence = sentence[1:]
        respaced_sentences.append(sentence)
    
    corpus = " ".join(respaced_sentences)
    scores = []
    for sentence in respaced_sentences:
        (R, _, _) = score(model, [sentence], [corpus])
        scores.append(R[0])

    indices = sentenceSelection(respaced_sentences, scores, reductionFactor)
    
    simplified = []
    for index in indices:
        simplified.append(respaced_sentences[index])
    
    simplified = " ".join(simplified)

    return simplified