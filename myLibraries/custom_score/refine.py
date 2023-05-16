from custom_score.utils import *
from custom_score.score import score 
from rouge_score import rouge_scorer
import bert_score
import pandas as pd
from scipy.stats import pearsonr
from colorama import Fore, Style
from datetime import datetime

import sys
sys.path.append(get_git_root())

from BARTScore.bart_score import BARTScorer

class Refiner:

    def __init__(self, corpus, gold, model, scorer=score, ratio=2, threshold=0.70, maxSpacing=10, printRange=range(0, 1)):
        """
        Constructor of the Refiner class. Aims at reducing the size and noise of a given independant list of documents.
        
        :param1 self (Refiner): Object to initialize.
        :param2 corpus (List): List of documents to simplify.
        :param3 gold (List): List of gold summaries to compare to the extractive summary created with the refiner.
        :param4 model (Any): Model used to compute scores and create sentence's ranking.
        :param5 ratio (float, int or array-like): Number determining how much the reference text will be shortened. 
        :param6 threshold (float): Number between 0 and 1 indicating the lowest acceptable quality when tuning the length of the summary.
        :param7 maxSpacing (int): Maximal number of adjacent space to be found and suppressed in the corpus.
        :param8 printRange (range): Range of corpus that should be displayed when the Refiner object in printed. 
        """
        self.corpus = corpus
        self.gold = gold
        self.processedCorpus = None
        self.model = model
        self.scorer = scorer
        self.ratio = ratio
        self.threshold = threshold
        self.ms = maxSpacing
        self.refined = None
        self.printRange = printRange
        self.selectedIndexes = None

    def refine(self, checkpoints=False, saveRate=50):
        """
        Return a reduced string computed using static embedding vectors similarity. Also denoises the data by removing superfluous elements such as "\n" or useless signs.

        :param1 self (Refiner): Refiner Object (see __init__ function for more details).
        :param2 checkpoints (bool): Indicates whether the refining should save partial outputs along computation to prevent from losing data in the context of a crash.
        :param3 saveRate (int): Only applicable id safe equals True. Specify the number of consicutive iterations after which a checkpoint should be created. 

        :output refined (string): refined version of the initial document.
        """
        self.refined = []
        self.selectedIndexes = []
        self.processedCorpus = []
        if checkpoints:
            iter = 0
            start = datetime.now()
            tempSave = False

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
            indices = None
            if type(self.ratio) == int or type(self.ratio) == float: 
                indices = sentenceSelection(respaced_sentences, scores, distances, self.ratio)
            else:
                for curRatio in sorted(self.ratio):
                    curIndices = sentenceSelection(respaced_sentences, scores, distances, curRatio)
                    subCurRefined = [respaced_sentences[i] for i in curIndices]
                    scoreOut = self.scorer(self.model, [" ".join(subCurRefined)], [indiv])
                    curScore = parseScore(scoreOut)
                    if curScore < self.threshold:
                        try:
                            indices = curBest
                        except:
                            indices = curIndices
                        finally:
                            break
                    else:
                        curBest = curIndices
                if indices is None:
                    indices = curIndices
            indices.sort()
            curRefined = []
            for index in indices:
                curRefined.append(respaced_sentences[index])
            curRefined = ". \n".join(curRefined) + "."
            self.selectedIndexes.append(indices)
            self.refined.append(curRefined)

            #checkpoint verification
            if checkpoints:
                if iter % saveRate == 0 and iter != 0:
                    stop = datetime.now()
                    partial_runtime = stop - start
                    self.save(runtime=partial_runtime)
                    tempSave = True
                iter += 1
        if checkpoints:
            stop = datetime.now()
            runtime = stop - start
            self.save(runtime=runtime, new=not(tempSave))

    def assess(self, start=0, stop=None, verbose=True):
        """
        Assesses quality of the refined corpus by computing Static BERTscore and Rouge-Score on the refined version compared to it's initial version.

        :param1 self (Refiner): Refiner Object (see __init__ function for more details).
        :param2 start (int): Starting index to assess.
        :param3 stop (int): Ending index to assess.
        :param4 verbose (Boolean): When put to True, assess results will be printed.

        :output (dict): Dictionnary containing both the scores of Static BERTScore, BERTScore, BARTScore and Rouge as well as their correlation.
        """
        assert self.refined != None, "refined corpus doesn't exists"
        
        if stop == None:
            stop = len(self.refined)
        subset_refined = self.refined[start:stop]
        subset_gold = self.gold[start:stop]

        #Static BERTScore computation
        scoreOut = score(self.model, subset_refined, subset_gold)
        customScore = [parseScore(curScore) for curScore in scoreOut]

        #Rouge-Score computation
        rougeScorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rougeScore = [rougeScorer.score(c, r) for c, r in zip(subset_gold, subset_refined)]

        #BERTScore computation
        with nostd():
            bertscore = bert_score.score(subset_refined, subset_gold, lang="en", verbose=0)

        #bartscore
        bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
        bartscore = bart_scorer.score(subset_refined, subset_gold, batch_size=4)

        #Data formating
        custom_R = [round(t, 2) for t in customScore]
        bertscore_R = [round(t.item(), 2) for t in bertscore[1]]
        bartscore = [round(t, 2) for t in bartscore]
        rouge1_R = [round(t['rouge1'][0], 2) for t in rougeScore]
        rouge2_R = [round(t['rouge2'][0], 2) for t in rougeScore]
        rougeL_R = [round(t['rougeL'][0], 2) for t in rougeScore]

        dfCustom = pd.DataFrame({'CBERT' : custom_R,
                                 'BERTScore' : bertscore_R,
                                 'BARTScore' : bartscore,
                                 'R-1' : rouge1_R,
                                 'R-2' : rouge2_R,
                                 'R-L' : rougeL_R
                                })

        #Correlation estimation
        pearsonCor_c_r1 = np.round(pearsonr(custom_R, rouge1_R), 2)
        pearsonCor_c_r2 = np.round(pearsonr(custom_R, rouge2_R), 2)
        pearsonCor_c_rl = np.round(pearsonr(custom_R, rougeL_R), 2)
        pearsonCor_bertscore_r1 = np.round(pearsonr(bertscore_R, rouge1_R), 2)
        pearsonCor_bertscore_r2 = np.round(pearsonr(bertscore_R, rouge2_R), 2)
        pearsonCor_bertscore_rl = np.round(pearsonr(bertscore_R, rougeL_R), 2)
        pearsonCor_bartscore_r1 = np.round(pearsonr(bartscore, rouge1_R), 2)
        pearsonCor_bartscore_r2 = np.round(pearsonr(bartscore, rouge2_R), 2)
        pearsonCor_bartscore_rl = np.round(pearsonr(bartscore, rougeL_R), 2)

        dfCor = pd.DataFrame({'pearson_CBERT_R-1' : pearsonCor_c_r1,
                              'pearson_CBERT_R-2' : pearsonCor_c_r2,
                              'pearson_CBERT_R-L' : pearsonCor_c_rl,
                              'pearson_BERT_R-1' : pearsonCor_bertscore_r1,
                              'pearson_BERT_R-2' : pearsonCor_bertscore_r2,
                              'pearson_BERT_R-l' : pearsonCor_bertscore_rl,
                              'pearson_BART_R-1' : pearsonCor_bartscore_r1,
                              'pearson_BART_R-2' : pearsonCor_bartscore_r2,
                              'pearson_BART_R-l' : pearsonCor_bartscore_rl}, index=["Pearson score", "p-value"])
        if verbose:
            printout = "Scores: \n"
            printout += dfCustom.to_string() + "\n\n"
            printout += "Correlations: \n"
            printout += dfCor.to_string()
            print(printout)

        return {"scores": dfCustom, "correlations": dfCor}
    
    def to_dataframe(self):
        """
        Transforms a Refiner object to a dataframe.

        :param1 self (Refiner): Refiner Object (see __init__ function for more details).

        :output output (DataFrame): DataFrame containing both the corpus and the refined texts of the Refiner class. 
        """
        output = pd.DataFrame({"text": self.corpus,
                               "summary": self.refined,
                               "processedText": [". ".join(c) for c in self.processedCorpus]})
        return output

    def save(self, runtime=None, new=True):
        """
        Saves Refiner output to a local folder.

        :param1 self (Refiner): Refiner Object (see __init__ function for more details).
        :param2 new (bool): Indicates if a new folder should be created. If false, output is append to the most recent ouput folder.
        """

        #evaluation
        start = 0
        stop = len(self.refined)
        assessement = self.assess(start=start, stop=stop)

        #mainDf = r.to_dataframe()
        scoreDf = assessement["scores"]
        corDf = assessement["correlations"]

        #write output
        main_folder_path = os.path.join(get_git_root(), r"myLibraries\refining_output")
        countfile_name = r"count.txt"
        if new:
            count = updateFileCount(os.path.join(main_folder_path, countfile_name))
        else:
            count = readFileCount(os.path.join(main_folder_path, countfile_name))

        current_path = os.path.join(main_folder_path, f"experimentation_{count}")
        try:
            os.mkdir(current_path)
        except FileExistsError:
            pass

        #mainDf.to_csv(os.path.join(current_path, "main.csv"))
        scoreDf.to_csv(os.path.join(current_path, "scores.csv"))
        corDf.to_csv(os.path.join(current_path, "correlations.csv"))
        with open(os.path.join(current_path, "runtimes.txt"), "w") as f:
            f.write(str(runtime))

    def __str__(self) -> str:
        """
        Summarizes Refiner object to a string.

        :param1 self (Refiner): Refiner Object (see __init__ function for more details).

        :output printout (string): Summarized informations about the refiner object.
        """

        printout = "--------REFINER OBJECT--------\n\n"
        printout += "Number of Documents : " + str(len(self.corpus)) + "\n"
        printout += "Corpus Avg Size     : " + str(int(np.average([len(x) for x in self.corpus]))+1) + "\n"
        printout += "Refined Avg Size    : " + str(int(np.average([len(x) for x in self.refined]))+1) + "\n"
        printout += "Ratio(s)            : " + str(self.ratio) + "\n"
        printout += "Threshold           : " + str(self.threshold) + "\n"
        printout += "Maximum Spacing     : " + str(self.ms) + "\n"
        
        self.printRange = self.printRange if self.printRange.start >= 0 and self.printRange.stop < len(self.processedCorpus) else range(0, len(self.processedCorpus))

        for index in self.printRange:
            printout += f"\nCorpus no.{index+1} : \n" + str(".\n".join([f"{Fore.LIGHTGREEN_EX}{self.processedCorpus[index][i]}{Style.RESET_ALL}"
                                                        if i in self.selectedIndexes[index]
                                                        else f"{Fore.RED}{self.processedCorpus[index][i]}{Style.RESET_ALL}"
                                                        for i in range(len(self.processedCorpus[index]))])) + "." + "\n"
        printout += "\n------------------------------"
        return printout