import numpy as np
from numpy.linalg import norm
from gensim.models import KeyedVectors
import pickle
from random import uniform
from contextlib import contextmanager
import sys, os, io
import git


def model_to_serialized(model, path):
    """
    Dumps a model to the specified path.

    :param1 model (Any): Variable of the system.
    :param2 path (string): Path to save the searialized model.
    """
    with open(path, 'wb') as f:
        pickle.dump(model, f)
        f.close()

def serialized_to_model(path):
    """
    Loads a serialized model saved to the specified path.

    :param1 path (string): Path where the serialized model is saved.

    :output model (Any): Variable loaded from the dumped memory.
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
        f.close()
    return model

def model_load(model, serialized=False):
    """
    Loads Keyed-vectors for the desired model.

    :param1 model (string): Name of the keyed-vector's model to import. 
                           "Word2Vec", "Fasttext" and "Glove" are supported.
    
    :output1 emb (dict): Dictionnary containing vectors and the word it's associated to.
    """
    assert(type(model) == str)

    #get ressource folder
    repo_path = os.path.join(get_git_root(), "myPaths.txt")
    if os.path.exists(repo_path):
        with open(repo_path, "r") as f:
            ressources_path = f.readlines()[1].strip()

    if model == "Word2Vec":
        if not serialized:
            try:
                wordvector_path = os.path.join(ressources_path, r'GoogleNews-vectors-negative300.bin.gz')
                emb = KeyedVectors.load_word2vec_format(wordvector_path, binary=True)
            except:
                return False
        else:
            try:
                serialized_wordvector_path = os.path.join(ressources_path, r'serialized_w2v.pkl')
                emb = serialized_to_model(serialized_wordvector_path)
            except:
                return False
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
            R.append(0.)

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

    #remove linebreaks and superfluous characters 
    clean = clean.replace("\n", " ")
    clean = clean.replace("-", "")
    clean = clean.replace("''", "")
    clean = clean.replace("``", "")
    clean = clean.replace('""', "")

    #remove overspacing
    spacing = "".join([" " for _ in range(maxSpacing)])
    for _ in range(maxSpacing-1):
        spacing = spacing[:-1]
        clean = clean.replace(spacing, " ")
    
    return clean

def sentenceSelection(corpus, scores, distances, ratio=2):
    """
    Returns a list of selected indices of sentence that will constituate the new corpus.

    :param1 corpus (list): List of sentences of the reference document.
    :param2 scores (list): List of the similarity scores of each sentence of the reference compared to the entire reference document.
    :param3 ratio (float or int): Number determining how much the reference text will be shortened. 

    :output selected_indexes (list): List of indexes of the initial corpus sentences that have been selected.
    """
    totalLength = len(" ".join(corpus))
    targetLength = int(totalLength/ratio)
    selected_indexes = []
    
    randomized_scores = [np.mean([curScore, uniform(0, 1)]) for curScore in scores]
    ranking = np.argsort(randomized_scores)[::-1]

    selectedLength = 0
    selected_indexes = []
    current_distance = lambda x: norm([distances[ranking[x]][i] for i in selected_indexes])
    cur = 0
    if targetLength != totalLength:
        while(selectedLength < targetLength):
            if cur == 0:
                selected_indexes.append(ranking[cur])
                selectedLength += len(corpus[ranking[cur]])
            else:
                updated_scores = [randomized_scores[i]*current_distance(i) for i in range(len(randomized_scores))]
                updated_ranking = np.argsort(updated_scores)[::-1]
                updated_ranking = [index for index in updated_ranking if index not in selected_indexes]
                selected_indexes.append(updated_ranking[0])
                selectedLength += len(corpus[updated_ranking[0]])
            cur += 1
    else:
        selected_indexes = ranking

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
        parsedScore = evalScore[0][position]
    return parsedScore

def parseDistances(distances):
    """
    Converts score to a value interpretable as distance (the greatest the value, the greatest the distance).

    :param1 distances (List): List of distances values for each sentences of a corpus with respect to all the others sentences.

    :output parsedDistances (List): List of converted distances. 
    """
    refDist = distances[0][0]
    toPositive = False 
    toInverse = False

    if refDist < 0:
        toPositive = True
    if refDist >=-1 and refDist <=1:
        toInverse = True
    
    parsedDistances = []
    for distance in distances:
        parsedDistance = []
        for value in distance:
            cur = value
            if toPositive:
                cur = -cur
            if toInverse:
                try:
                    cur = 1/cur
                except ZeroDivisionError:
                    cur = 1
            parsedDistance.append(cur)
        parsedDistances.append(parsedDistance)
    return parsedDistances

def isIncreasingRange(r):
    assert type(r) == range, "Error: not a range object"
    if r.start <= r.stop and r.step > 0:
        return True
    else:
        return False

@contextmanager
def nostd():
    """
    Decorator supressing console output. Source : https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto
    """
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    sys.stdout = io.BytesIO()
    sys.stderr = io.BytesIO()
    yield
    sys.stdout = save_stdout
    sys.stderr = save_stderr

def get_git_root():
    git_repo = git.Repo(os.path.abspath(__file__), search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root