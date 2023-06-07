from MARScore.utils import *
import os

import sys
sys.path.append(get_git_root())
sys.path.append(os.path.join(get_git_root(), "myLibraries"))

from custom_score.score import score
from custom_score.utils import model_load
from rouge_score import rouge_scorer
import bert_score
import hdbscan
import pandas as pd
from scipy.stats import pearsonr
from BARTScore.bart_score import BARTScorer
from colorama import Fore, Style

class MARSCore():
    def __init__(self, corpus, gold, model=BertModel.from_pretrained('bert-base-uncased', 
                                                               output_hidden_states=True), 
                               tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')) -> None:
        """
        Constructor of the MARScore class.

        :param1 corpus (list): List of texts to summarize.
        :param2 model (transformer): Transformer model used compute dynamic embeddings.
        :param3 tokenizer (transformer) Transformer used to create token from a plain text. 
        """
        self.corpus = corpus
        self.gold = gold
        self.summaries = []
        self.model = model
        self.tokenizer = tokenizer
    
    def compute(self):
        for indiv in self.corpus:
            #creation of embeddings
            o, l = tokenizeCorpus(indiv)
            v = vectorizeCorpus(o)
            v, l = cleanAll(v, l)

            #clusterization
            clusterer = hdbscan.HDBSCAN()
            clusterer.fit(v)
            clabels = clusterer.labels_

            #TF calculation
            tf_values = tf(l)
            clusters_tf_values = clusters_tf(tf_values, l, clabels)

            #ILP computation
            check = to_ilp_format(l, clabels, clusters_tf_values)
            root = get_git_root()
            dirpath = os.path.join(root, "myLibraries\MARScore_output")
            os.system(f'glpsol --tmlim 100 --lp "{os.path.join(dirpath, "ilp_in.ilp")}" -o "{os.path.join(dirpath, "ilp_out.sol")}"')
            selected = readILP()

            #summaries construction
            sentences = indiv.split(".")
            sentences.pop()
            sum_sentences = []
            for i, value in enumerate(selected):
                if value == 1:
                    sum_sentences.append(sentences[i]+".")
            self.summaries.append(" ".join(sum_sentences))

    def assess(self, start=0, stop=None, verbose=True):
        """
        Assesses quality of the refined corpus by computing Static BERTscore and Rouge-Score on the refined version compared to it's initial version.

        :param1 self (Refiner): Refiner Object (see __init__ function for more details).
        :param2 start (int): Starting index to assess.
        :param3 stop (int): Ending index to assess.
        :param4 verbose (Boolean): When put to True, assess results will be printed.

        :output (dict): Dictionnary containing both the scores of Static BERTScore, BERTScore, BARTScore and Rouge as well as their correlation.
        """
        assert self.summaries != None, "refined corpus doesn't exists"
        
        if stop == None:
            stop = len(self.summaries)
        subset_summaries = self.summaries[start:stop]
        subset_gold = self.gold[start:stop]

        #Static BERTScore computation
        w2v = model_load("Word2Vec", True)
        customScore = score(w2v, subset_summaries, subset_gold)
        #customScore = [parseScore(curScore) for curScore in scoreOut]

        #Rouge-Score computation
        rougeScorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rougeScore = [rougeScorer.score(c, r) for c, r in zip(subset_gold, subset_summaries)]

        #BERTScore computation
        with nostd():
            bertscore = bert_score.score(subset_summaries, subset_gold, lang="en", verbose=0)
        #bartscore
        bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
        bartscore = bart_scorer.score(subset_summaries, subset_gold, batch_size=4)

        #Data formating
        custom_R = [round(t[0], 2) for t in customScore]
        custom_P = [round(t[1], 2) for t in customScore]
        custom_F = [round(t[2], 2) for t in customScore]
        bertscore_R = [round(t.item(), 2) for t in bertscore[1]]
        bertscore_P = [round(t.item(), 2) for t in bertscore[0]]
        bertscore_F = [round(t.item(), 2) for t in bertscore[2]]
        bartscore = [round(t, 2) for t in bartscore]
        rouge1_R = [round(t['rouge1'][0], 2) for t in rougeScore]
        rouge2_R = [round(t['rouge2'][0], 2) for t in rougeScore]
        rougeL_R = [round(t['rougeL'][0], 2) for t in rougeScore]

        dfCustom = pd.DataFrame({'CBERT-R' : custom_R,
                                 'CBERT-P' : custom_P,
                                 'CBERT-F' : custom_F,
                                 'BERTScore-R' : bertscore_R,
                                 'BERTScore-P' : bertscore_P,
                                 'BERTScore-F' : bertscore_F,
                                 'BARTScore' : bartscore,
                                 'R-1' : rouge1_R,
                                 'R-2' : rouge2_R,
                                 'R-L' : rougeL_R
                                })

        #Correlation estimation
        pearsonCor_c_r1 = np.round(pearsonr(custom_F, rouge1_R), 2)
        pearsonCor_c_r2 = np.round(pearsonr(custom_F, rouge2_R), 2)
        pearsonCor_c_rl = np.round(pearsonr(custom_F, rougeL_R), 2)
        pearsonCor_bertscore_r1 = np.round(pearsonr(bertscore_F, rouge1_R), 2)
        pearsonCor_bertscore_r2 = np.round(pearsonr(bertscore_F, rouge2_R), 2)
        pearsonCor_bertscore_rl = np.round(pearsonr(bertscore_F, rougeL_R), 2)
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

    def __str__(self) -> str:
        """
        Summarizes MARScore object to a string.

        :param1 self (MARScore): MARScore Object (see __init__ function for more details).

        :output printout (string): Summarized informations about the MARSCore object.
        """

        printout = "--------MARScore OBJECT--------\n\n"
        printout += "Number of Documents : " + str(len(self.corpus)) + "\n"
        printout += "Corpus Avg Size     : " + str(int(np.average([len(x) for x in self.corpus]))+1) + "\n"
        printout += "Refined Avg Size    : " + str(int(np.average([len(x) for x in self.summaries]))+1) + "\n"

        """    
        self.printRange = self.printRange if self.printRange.start >= 0 and self.printRange.stop < len(self.processedCorpus) else range(0, len(self.processedCorpus))
    
        for index in self.printRange:
            printout += f"\nCorpus no.{index+1} : {str(self.scores[index]*100)+'%' if self.scores != [] and self.scores != -1 else ''}\n" + str(".\n".join([f"{Fore.LIGHTGREEN_EX}{self.processedCorpus[index][i]}{Style.RESET_ALL}"
                                                        if i in self.selectedIndexes[index]
                                                        else f"{Fore.RED}{self.processedCorpus[index][i]}{Style.RESET_ALL}"
                                                        for i in range(len(self.processedCorpus[index]))])) + "." + "\n"
        """
        printout += "\n-------------------------------"
        return printout