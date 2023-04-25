import numpy as np
from numpy.linalg import norm
from gensim.models import KeyedVectors
import pickle
from random import uniform


def model_to_serialized(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
        f.close()

def serialized_to_model(path):
    with open(path, 'rb') as f:
        w2v = pickle.load(f)
        f.close()
    return w2v

def model_load(model, serialized=False):
    """
    Loads Keyed-vectors for the desired model.

    :param1 model (string): Name of the keyed-vector's model to import. 
                           "Word2Vec", "Fasttext" and "Glove" are supported.
    
    :output1 emb (dict): Dictionnary containing vectors and the word it's associated to.
    """
    assert(type(model) == str)
    if model == "Word2Vec":
        if not serialized:
            try:
                wordvector_path = r'D:\COURS\A4\S8\Stage\Documents\Supervised-Learning-using-Unsupervised-Learning-Metrics-in-the-absence-of-Annotated-Data\custom_BERTScore\GoogleNews-vectors-negative300.bin.gz'
                emb = KeyedVectors.load_word2vec_format(wordvector_path, binary=True)
            except:
                wordvector_path = r'D:\COURS\A4\S8 - ESILV\Stage\Work\Models\GoogleNews-vectors-negative300.bin.gz'
                emb = KeyedVectors.load_word2vec_format(wordvector_path, binary=True)
        else:
            try:
                serialized_wordvector_path = r'D:\COURS\A4\S8\Stage\Documents\Models\serialized_w2v.pkl'
                emb = serialized_to_model(serialized_wordvector_path)
            except:
                serialized_wordvector_path = r'D:\COURS\A4\S8 - ESILV\Stage\Work\Models\serialized_w2v.pkl'
                emb = serialized_to_model(serialized_wordvector_path)
    else:
        print("Model not supported yet")
    return emb

def encode(corpus, model):
    """
    Encodes words into n-dimensional vectors depending on the model.

    :param1 corpus (list): List of sentences.
    :param2 model (dict): Dictionnary of the embedding.

    :output1 encoded_corpus (list): Encoded list of sentences.
    :output2 n_unknown (int): Number of unknown token found in the corpus.
    """
    encoded_corpus = []
    unknown = 0
    for sentence in corpus:
        encoded_sentence = []
        for word in sentence.split(" "):
            try:
                encoded_sentence.append(model[word])
            except:
                unknown += 1
        encoded_corpus.append(encoded_sentence)
    return np.array(encoded_corpus, dtype=object), unknown

def similarityCandToRef(references, candidates):
    """
    Computes cosine similarity for every reference with respect to each candidate.

    :param1 references (list): List of reference sentences.
    :param2 candidates (list): List of candidate sentences.

    :output1 all_proximities (list): List of similarity matrix between each reference/candidate couple.  
    """
    proximity = lambda x, y: (np.matmul(np.transpose(x), y))/(norm(x)*norm(y))

    all_proximities = []

    for candidate, reference in zip(candidates, references):
        proximities = []
        for c_word in candidate:
            sub_proximities = []
            for r_word in reference:
                sub_proximities.append(proximity(r_word, c_word))
            proximities.append(sub_proximities)
        all_proximities.append(proximities)
    return all_proximities

def similarityRefToCand(references, candidates):
    """
    Computes cosine similarity for every reference with respect to each candidate.

    :param1 references (list): List of reference sentences.
    :param2 candidates (list): List of candidate sentences.

    :output1 all_proximities (list): List of similarity matrix between each reference/candidate couple.  
    """
    proximity = lambda x, y: (np.matmul(np.transpose(x), y))/(norm(x)*norm(y))

    all_proximities = []

    for candidate, reference in zip(candidates, references):
        proximities = []
        for r_word in reference:
            sub_proximities = []
            for c_word in candidate:
                sub_proximities.append(proximity(r_word, c_word))
            proximities.append(sub_proximities)
        all_proximities.append(proximities)
    return all_proximities

def computeMetrics(refToCand, candToRef, references, candidates):
    """
    Calculates R, P and F measures for a given corpus

    :param1 refToCand (list): List of similarity matrix between each reference/candidate couple.
    :param2 candToRef (list): List of similarity matrix between each reference/candidate couple.
    :param3 references (list): List of reference sentences.
    :param4 candidates (list): List of candidate sentences.

    :output (tuple): Tuple containing R, P and F for the current corpus.
    """

    # R computation
    fullSum = []
    for individualSimilarity in refToCand:
        currentSum = 0
        for row in individualSimilarity:
            currentSum += np.max(row)
        fullSum.append(currentSum)
    R = []
    for sum, reference in zip(fullSum, references):
        try:
            R.append((1/len(reference))*sum)
        except ZeroDivisionError:
            R.append(0)

    # P computation
    fullSum = []
    for individualSimilarity in candToRef:
        currentSum = 0
        for row in individualSimilarity:
            currentSum += np.max(row)
        fullSum.append(currentSum)
    P = []
    for sum, candidate in zip(fullSum, candidates):
        try:
            P.append((1/len(candidate))*sum)
        except ZeroDivisionError:
            P.append(0.)
    
    # F computation
    F = []
    for r, p in zip(R, P):
        try:
            f = 2*((p*r)/(p+r))
        except ZeroDivisionError:
            f = 0
        F.append(f)
    
    return (R, P, F)

def computeIdf(corpus):
    """
    Calculates IDF all words of a corpus
    Inspired by : https://www.freecodecamp.org/news/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3/

    :param1 corpus (String): Reference document.

    :output1 idfDict (dict): IDf dictionnary for a given corpus.
    """

    idfDict = {}
    splitCorpus = corpus.split(" ")
    N = len(splitCorpus)
    #idfDict = dict.fromkeys(set(splitCorpus), 0)
    for word in splitCorpus:
        wordFreq = splitCorpus.count(word)
        idfDict[word.lower()] = -np.log(1/N * wordFreq) 
    return idfDict

def getIdf(idfDict, word):
    """
    Returns the IDF of a word in a given corpus.

    :param1 idfDict (dict): Dictionnary of all IDFs of a document.
    :param2 word (string): Word whose IDF is desired.

    :output idf (float): IDF of the desired word.
    """
    try:
        idf = idfDict[word.lower()]
    except:
        idf = 1.

    return idf

def computeMetricsWithIdf(refToCand, candToRef, referencesWords, candidatesWords, allIdfDicts):
    """
    Calculates R, P and F measures for a given corpus using an IDF weighting.

    :param1 refToCand (list): List of similarity matrix between each reference/candidate couple.
    :param2 candToRef (list): List of similarity matrix between each reference/candidate couple.
    :param3 references (list): List of reference sentences.
    :param4 candidates (list): List of candidate sentences.

    :output (tuple): Tuple containing R, P and F for the current corpus.
    """
    # R computation
    fullSum = []
    fullIdfSum = []
    for individualSimilarity, candidate, idfDict in zip(refToCand, candidatesWords, allIdfDicts):
        currentSum = 0
        currentIdfSum = 0
        for row, word in zip(individualSimilarity, candidate):
            currentMax = np.max(row)
            currentIdf = getIdf(idfDict, word)
            currentIdfSum += currentIdf
            currentSum += (currentMax * currentIdf)
        fullIdfSum.append(currentIdfSum)
        fullSum.append(currentSum)
    R = []
    for sum, idfSum in zip(fullSum, fullIdfSum):
        R.append((1/idfSum)*sum)

    # P computation
    fullSum = []
    fullIdfSum = []
    for individualSimilarity, reference, idfDict in zip(candToRef, referencesWords, allIdfDicts):
        currentSum = 0
        currentIdfSum = 0
        for row, word in zip(individualSimilarity, reference):
            currentMax = np.max(row)
            currentIdf = getIdf(idfDict, word)
            currentIdfSum += currentIdf
            currentSum += (currentMax * currentIdf)
        fullIdfSum.append(currentIdfSum)
        fullSum.append(currentSum)
    P = []
    for sum, idfSum in zip(fullSum, fullIdfSum):
        P.append((1/idfSum)*sum)

    # F computation
    F = []
    for r, p in zip(R, P):
        f = 2*((p*r)/(p+r))
        F.append(f)

    return (R, P, F)

def cleanString(string, maxSpacing=10):
    """
    :param1 string (string): Initial corpus
    :param2 maxSpacing (int): Maximal number of adjacent space to be found and suppressed in the corpus.

    :output clean (string): Cleansed corpus
    """

    clean = string[:]

    #remove linebreaks  
    clean = clean.replace("\n", " ")
    clean = clean.replace("-", "")

    #remove surplus spacing
    spacing = "".join([" " for _ in range(maxSpacing)])
    for _ in range(maxSpacing-1):
        spacing = spacing[:-1]
        clean = clean.replace(spacing, " ")
    
    return clean

def sentenceSelection(corpus, scores, reductionFactor=2):
    """
    Returns a list of selected indices of sentence that will constituate the new corpus.

    :param1 corpus (list): List of sentences of the reference document.
    :param2 scores (list): List of the similarity scores of each sentence of the reference compared to the entire reference document.
    :param3 reductionFactor (float or int): Number determining how much the reference text will be shortened. 

    :output selected_indexes (list): List of indexes of the initial corpus sentences that have been selected.
    """
    totalLength = len(corpus)
    targetLength = int(totalLength/reductionFactor)
    selected_indexes = []
    
    randomized_scores = [np.mean([curScore, uniform(0, 1)]) for curScore in scores]
    ranking = np.argsort(randomized_scores)[::-1]
    selected_indexes = ranking[:targetLength]

    return selected_indexes

def parseScore(evalScore, position=0):
    """
    Extracted the metric of interrest from the output structure of the scorer function.

    :param1 evalScore (unknown): Output of the scorer function.
    
    :output parsedScore (float): Extracted score value. 
    """
    castedEvalScore = np.array(evalScore)
    if castedEvalScore.ndim == 0:
        parsedScore = evalScore
    if castedEvalScore.ndim == 1:
        parsedScore = evalScore[0]
    if castedEvalScore.ndim == 2:
        parsedScore = evalScore[0][position] #considering recall is the first element
    return parsedScore