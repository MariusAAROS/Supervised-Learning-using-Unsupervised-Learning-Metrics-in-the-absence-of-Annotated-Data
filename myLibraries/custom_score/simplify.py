from .utils import *
from .score import score 
from rouge_score import rouge_scorer


class Simplifier:

    def __init__(self, corpus, model, scorer=score, reductionFactor=2, maxSpacing=10):
        self.corpus = corpus
        self.model = model
        self.scorer = scorer
        self.rf = reductionFactor
        self.ms = maxSpacing
        self.simplified = None

    def simplify(self):
        """
        Return a reduced string computed using static embedding vectors similarity. Also denoises the data by removing superfluous elements such as "\n" or useless signs.
    
        :param1 reference (string): Document to simplify.
        :param2 model (dict): Dictionnary of keyed-vectors.
        :param3 reductionFactor (float or int): Number determining how much the reference text will be shortened. 
        :param4 maxSpacing (int): Maximal number of adjacent space to be found and suppressed in the corpus.

        :output simplified (string): Simplified version of the initial document.
        """
        self.simplified = []
        for indiv in self.corpus:
            #preprocess corpus
            clean = cleanString(indiv, self.ms)
            sentences = clean.split(".")
            sentences.pop()
            respaced_sentences = []
            for sentence in sentences:
                if sentence[0] == " ":
                    sentence = sentence[1:]
                respaced_sentences.append(sentence)
            corpus = " ".join(respaced_sentences)

            #compute ranking
            scores = []
            for sentence in respaced_sentences:
                scoreOut = score(self.model, [sentence], [indiv])
                R = parseScore(scoreOut)
                scores.append(R)

            #selection of best individuals
            indices = sentenceSelection(respaced_sentences, scores, self.rf)
            
            curSimplified = []
            for index in indices:
                curSimplified.append(respaced_sentences[index])
            
            curSimplified = " ".join(curSimplified)
            self.simplified.append(curSimplified)

        def assess(self):
            assert self.simplified != None, "simplified corpus doesn't exists"
