from .utils import *
from .score import score 
from rouge_score import rouge_scorer
import pandas as pd
from scipy.stats import pearsonr
from numpy.linalg import norm


class Refiner:

    def __init__(self, corpus, model, scorer=score, reductionFactor=2, maxSpacing=10):
        """
        Constructor of the Refiner class. Aims at reducing the size and noise of a given independant list of documents.
        
        :param1 self (Refiner): Object to initialize.
        :param2 corpus (List): List of documents to simplify.
        :param3 model (Any): Model used to compute scores and create sentence's ranking.
        :param4 reductionFactor (float or int): Number determining how much the reference text will be shortened. 
        :param5 maxSpacing (int): Maximal number of adjacent space to be found and suppressed in the corpus.
        """
        self.corpus = corpus
        self.model = model
        self.scorer = scorer
        self.rf = reductionFactor
        self.ms = maxSpacing
        self.refined = None

    def refine(self):
        """
        Return a reduced string computed using static embedding vectors similarity. Also denoises the data by removing superfluous elements such as "\n" or useless signs.

        :param1 self (Refiner): Refiner Object (see __init__ function for more details).

        :output refined (string): refined version of the initial document.
        """
        self.refined = []
        for indiv in self.corpus:
            #preprocess corpus
            clean = cleanString(indiv, self.ms)
            sentences = clean.split(".")
            sentences.pop()
            temp = []
            for sentence in sentences: 
                if sentence != None and sentence != "":
                    temp.append(sentence)
            sentences = temp
            respaced_sentences = []
            for sentence in sentences:
                if sentence[0] == " ":
                    sentence = sentence[1:]
                respaced_sentences.append(sentence)
            corpus = " ".join(respaced_sentences)

            #compute ranking
            scores = []
            for sentence in respaced_sentences:
                scoreOut = self.scorer(self.model, [sentence], [indiv])
                R = parseScore(scoreOut)
                scores.append(R)
            
            #compute distances
            distances = []
            for x in range(len(respaced_sentences)):
                distance = []
                for y in range(len(respaced_sentences)):
                    if x != y:
                        try:
                            scoreOut = self.scorer(self.model, [respaced_sentences[x]], [respaced_sentences[y]])
                            curDistance = parseScore(scoreOut)
                        except:
                            curDistance = -1
                    else:
                        curDistance = 1
                    distance.append(curDistance)
                distances.append(distance)
            distances = parseDistances(distances)

            #selection of best individuals
            indices = sentenceSelection(respaced_sentences, scores, distances, self.rf)
            indices.sort()
            curRefined = []
            for index in indices:
                curRefined.append(respaced_sentences[index])
            
            curRefined = " ".join(curRefined)
            self.refined.append(curRefined)

    def assess(self, verbose=True):
        """
        Assesses quality of the refined corpus by computing Static BERTscore and Rouge-Score on the refined version compared to it's initial version.

        :param1 self (Refiner): Refiner Object (see __init__ function for more details).
        :param2 verbose (Boolean): When put to true, assess results will be printed.

        :output (dict): Dictionnary containing both the scores of Static BERTScore and Rouge as well as their correlation
        """
        assert self.refined != None, "refined corpus doesn't exists"

        #Static BERTScore computation
        scoreOut = self.scorer(self.model, self.refined, self.corpus)
        customScore = [parseScore(curScore) for curScore in scoreOut]

        #Rouge-Score computation
        rougeScorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        rougeScore = [rougeScorer.score(s, c) for s, c in zip(self.refined, self.corpus)]

        #Data formating
        custom_R = [round(t, 2) for t in customScore]
        rouge1_R = [round(t['rouge1'][0], 2) for t in rougeScore]
        rougeL_R = [round(t['rougeL'][0], 2) for t in rougeScore]

        dfCustom = pd.DataFrame({'CBERT' : custom_R,
                                'R-1' : rouge1_R,
                                'R-L' : rougeL_R
                                })

        #Correlation estimation
        pearsonCor_c_r1 = np.round(pearsonr(custom_R, rouge1_R), 2)
        pearsonCor_c_rl = np.round(pearsonr(custom_R, rougeL_R), 2)

        dfCor = pd.DataFrame({'pearson_CBERT_R-1' : pearsonCor_c_r1,
                            'pearson_CBERT_R-L' : pearsonCor_c_rl}, index=["Pearson score", "p-value"])

        return {"scores": dfCustom, "correlations": dfCor}
    
    def __str__(self) -> str:
        printout = "--------REFINER OBJECT--------\n"
        printout += "Number of Documents : " + str(len(self.corpus)) + "\n"
        printout += "Corpus Avg Size     : " + str(int(np.average([len(x) for x in self.corpus]))+1) + "\n"
        printout += "Refined Avg Size    : " + str(int(np.average([len(x) for x in self.refined]))+1) + "\n"
        printout += "Reduction Factor    : " + str(self.rf) + "\n"
        printout += "Maximum Spacing     : " + str(self.ms) + "\n"
        printout += "------------------------------"
        return printout