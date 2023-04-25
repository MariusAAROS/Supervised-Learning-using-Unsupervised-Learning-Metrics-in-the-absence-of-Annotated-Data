from .utils import *
from .score import score 


class Simplifier:

    def __init__(self, corpus, model, reductionFactor=2, maxSpacing=10):
        self.corpus = corpus
        self.model = model
        self.rf = reductionFactor
        self.ms = maxSpacing

    def simplify(self):
        """
        Return a reduced string computed using static embedding vectors similarity. Also denoises the data by removing superfluous elements such as "\n" or useless signs.
    
        :param1 reference (string): Document to simplify.
        :param2 model (dict): Dictionnary of keyed-vectors.
        :param3 reductionFactor (float or int): Number determining how much the reference text will be shortened. 
        :param4 maxSpacing (int): Maximal number of adjacent space to be found and suppressed in the corpus.

        :output simplified (string): Simplified version of the initial document.
        """

        #preprocess corpus
        clean = cleanString(self.corpus, self.ms)
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
            (R, _, _) = score(self.model, [sentence], [corpus])
            scores.append(R[0])

        indices = sentenceSelection(respaced_sentences, scores, self.rf)
        
        simplified = []
        for index in indices:
            simplified.append(respaced_sentences[index])
        
        simplified = " ".join(simplified)
        self.simplified = simplified