from .utils import *
from .score import score 
from rouge_score import rouge_scorer
import bert_score
import pandas as pd
from scipy.stats import pearsonr
from colorama import Fore, Style
import contextlib


class Refiner:

    def __init__(self, corpus, model, scorer=score, ratio=2, maxSpacing=10, printRange=range(0, 1)):
        """
        Constructor of the Refiner class. Aims at reducing the size and noise of a given independant list of documents.
        
        :param1 self (Refiner): Object to initialize.
        :param2 corpus (List): List of documents to simplify.
        :param3 model (Any): Model used to compute scores and create sentence's ranking.
        :param4 ratio (float or int): Number determining how much the reference text will be shortened. 
        :param5 maxSpacing (int): Maximal number of adjacent space to be found and suppressed in the corpus.
        :param6 printRange (range): Range of corpus that should be displayed when the Refiner object in printed. 
        """
        self.corpus = corpus
        self.processedCorpus = None
        self.model = model
        self.scorer = scorer
        self.ratio = ratio
        self.ms = maxSpacing
        self.refined = None
        self.printRange = printRange
        self.selectedIndexes = None

    def refine(self):
        """
        Return a reduced string computed using static embedding vectors similarity. Also denoises the data by removing superfluous elements such as "\n" or useless signs.

        :param1 self (Refiner): Refiner Object (see __init__ function for more details).

        :output refined (string): refined version of the initial document.
        """
        self.refined = []
        self.selectedIndexes = []
        self.processedCorpus = []
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
            self.processedCorpus.append(respaced_sentences)

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
            indices = sentenceSelection(respaced_sentences, scores, distances, self.ratio)
            indices.sort()
            curRefined = []
            for index in indices:
                curRefined.append(respaced_sentences[index])
            
            curRefined = ". \n".join(curRefined) + "."
            self.selectedIndexes.append(indices)
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
        rougeScore = [rougeScorer.score(c, r) for c, r in zip(self.corpus, self.refined)]

        #BERTScore computation
        
        #bertscore = [bert_score.score([r], [c], lang="en", verbose=0) for c, r in zip(self.corpus, self.refined)]
        with nostd():
            bertscore = bert_score.score(self.refined, self.corpus.to_list(), lang="en", verbose=0)

        #Data formating
        custom_R = [round(t, 2) for t in customScore]
        bertscore_R = [round(t.item(), 2) for t in bertscore[1]]
        rouge1_R = [round(t['rouge1'][0], 2) for t in rougeScore]
        rougeL_R = [round(t['rougeL'][0], 2) for t in rougeScore]

        dfCustom = pd.DataFrame({'CBERT' : custom_R,
                                 'BERTScore' : bertscore_R,
                                'R-1' : rouge1_R,
                                'R-L' : rougeL_R
                                })

        #Correlation estimation
        pearsonCor_c_r1 = np.round(pearsonr(custom_R, rouge1_R), 2)
        pearsonCor_c_rl = np.round(pearsonr(custom_R, rougeL_R), 2)
        pearsonCor_bertscore_r1 = np.round(pearsonr(bertscore_R, rouge1_R), 2)
        pearsonCor_bertscore_rl = np.round(pearsonr(bertscore_R, rougeL_R), 2)

        dfCor = pd.DataFrame({'pearson_CBERT_R-1' : pearsonCor_c_r1,
                              'pearson_CBERT_R-L' : pearsonCor_c_rl,
                              'pearson_BERT_R-1' : pearsonCor_bertscore_r1,
                              'pearson_BERT_R-l' : pearsonCor_bertscore_rl}, index=["Pearson score", "p-value"])
        if verbose:
            printout = "Scores: \n"
            printout += dfCustom.to_string() + "\n\n"
            printout += "Correlations: \n"
            printout += dfCor.to_string()
            print(printout)

        return {"scores": dfCustom, "correlations": dfCor}
    
    def __str__(self) -> str:
        printout = "--------REFINER OBJECT--------\n\n"
        printout += "Number of Documents : " + str(len(self.corpus)) + "\n"
        printout += "Corpus Avg Size     : " + str(int(np.average([len(x) for x in self.corpus]))+1) + "\n"
        printout += "Refined Avg Size    : " + str(int(np.average([len(x) for x in self.refined]))+1) + "\n"
        printout += "Reduction Factor    : " + str(self.ratio) + "\n"
        printout += "Maximum Spacing     : " + str(self.ms) + "\n"
        
        self.printRange = self.printRange if self.printRange.start >= 0 and self.printRange.stop < len(self.processedCorpus) else range(0, len(self.processedCorpus))

        for index in self.printRange:
            printout += f"\nCorpus no.{index+1} : \n" + str(".\n".join([f"{Fore.LIGHTGREEN_EX}{self.processedCorpus[index][i]}{Style.RESET_ALL}"
                                                        if i in self.selectedIndexes[index]
                                                        else f"{Fore.RED}{self.processedCorpus[index][i]}{Style.RESET_ALL}"
                                                        for i in range(len(self.processedCorpus[index]))])) + "." + "\n"
        printout += "\n------------------------------"
        return printout