import numpy as np
from numpy.linalg import norm
from gensim.models import KeyedVectors


def model_load(model):
    """
    Loads Keyed-vectors for the desired model.

    :param1 model (string): Name of the keyed-vector's model to import. 
                           "Word2Vec", "Fasttext" and "Glove" are supported.
    
    :output1 emb (dict): Dictionnary containing vectors and the word it's associated to.
    """
    assert(type(model) == str)
    if model == "Word2Vec":
        wordvector_path = r'D:\COURS\A4\S8\Stage\Documents\Supervised-Learning-using-Unsupervised-Learning-Metrics-in-the-absence-of-Annotated-Data\custom_BERTScore\GoogleNews-vectors-negative300.bin.gz'
        emb = KeyedVectors.load_word2vec_format(wordvector_path, binary=True)
    else:
        print("Model not currently supported")
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

def SimilarityCandToRef(references, candidates):
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

def SimilarityRefToCand(references, candidates):
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
    for individualSimilarity in candToRef:
        currentSum = 0
        for row in individualSimilarity:
            currentSum += np.max(row)
        fullSum.append(currentSum)
    R = []
    for sum, reference in zip(fullSum, references):
        R.append((1/len(reference))*sum)

    # P computation
    fullSum = []
    for individualSimilarity in refToCand:
        currentSum = 0
        for row in individualSimilarity:
            currentSum += np.max(row)
        fullSum.append(currentSum)
    P = []
    for sum, candidate in zip(fullSum, candidates):
        P.append((1/len(candidate))*sum)
    
    # F computation
    F = []
    for r, p in zip(R, P):
        f = 2*((p*r)/(p+r))
        F.append(f)
    
    return (R, P, F)