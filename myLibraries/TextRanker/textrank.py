import spacy
import pytextrank
from operator import itemgetter
from math import sqrt
import bert_score
from rouge_score import rouge_scorer
from scipy.stats import pearsonr
from TextRanker.utils import *
import pandas as pd
import numpy as np

import os
import sys
sys.path.append(get_git_root())
sys.path.append(os.path.join(get_git_root(), "myLibraries"))

from BARTScore.bart_score import BARTScorer
from custom_score.score import score
from custom_score.utils import model_load



class TextRanker():
    def __init__(self, corpus, gold, model="en_core_web_sm", expe_params=None) -> None:
        self.corpus = corpus
        self.gold = gold
        self.model = model
        self.summaries = []
        self.expe_params = expe_params
    def compute(self, limit_phrases=4, limit_sentences=2):
        nlp = spacy.load(self.model)
        nlp.add_pipe("textrank", last=True)

        for text in self.corpus:
            doc = nlp(text)
            sent_bounds = [[s.start, s.end, set([])] for s in doc.sents]

            phrase_id = 0
            unit_vector = []

            for p in doc._.phrases:
                unit_vector.append(p.rank)
                for chunk in p.chunks:
                    for sent_start, sent_end, sent_vector in sent_bounds:
                        if chunk.start >= sent_start and chunk.end <= sent_end:
                            sent_vector.add(phrase_id)
                            break
                phrase_id += 1
                if phrase_id == limit_phrases:
                    break
            
            sum_ranks = sum(unit_vector)
            unit_vector = [rank/sum_ranks for rank in unit_vector]
            sent_rank = {}
            sent_id = 0

            for sent_start, sent_end, sent_vector in sent_bounds:
                sum_sq = 0.0
                for phrase_id in range(len(unit_vector)):
                    if phrase_id not in sent_vector:
                        sum_sq += unit_vector[phrase_id]**2.0

                sent_rank[sent_id] = sqrt(sum_sq)
                sent_id += 1

            sent_text = {}
            sent_id = 0

            for sent in doc.sents:
                sent_text[sent_id] = sent.text
                sent_id += 1

            num_sent = 0

            result = ""
            for sent_id, rank in sorted(sent_rank.items(), key=itemgetter(1)):
                result += sent_text[sent_id] + " "
                num_sent += 1
                if num_sent == limit_sentences:
                    break
            self.summaries.append(result[:-1])

    def assess(self, start=0, stop=None, verbose=True):
        """
        Assesses quality of the refined corpus by computing Static BERTscore and Rouge-Score on the refined version compared to it's initial version.

        :param1 self (MARScore): MARScore Object (see __init__ function for more details).
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
        #score storing
        self.scores = bertscore_F

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
    
    def save(self, runtime=None, new=True):
            """
            Saves MARScore output to a local folder.

            :param1 self (MARScore): Refiner Object (see __init__ function for more details).
            :param2 new (bool): Indicates if a new folder should be created. If false, output is append to the most recent ouput folder.
            """
            #evaluation
            start = 0
            stop = len(self.summaries)
            assessement = self.assess(start=start, stop=stop)

            #mainDf = self.to_dataframe()
            scoreDf = assessement["scores"]
            corDf = assessement["correlations"]

            #write statistics
            if self.expe_params == None:
                main_folder_path = os.path.join(get_git_root(), r"myLibraries\MARScore_output\textrank_compare\result")
            elif "shuffled" in self.expe_params.keys():
                if self.expe_params["shuffled"]:
                    main_folder_path = os.path.join(get_git_root(), r"myLibraries\MARScore_output\textrank_compare\shuffled")
                else:
                    main_folder_path = os.path.join(get_git_root(), r"myLibraries\MARScore_output\textrank_compare\regular")
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
            
            #write summaries
            try:
                os.mkdir(os.path.join(current_path, f"summaries"))
            except FileExistsError:
                pass
            
            sum_path = os.path.join(current_path, f"summaries")
            

            for i, summary in enumerate(self.summaries):
                if new:
                    arg = "w"
                else:
                    arg = "a"
                with open(os.path.join(sum_path, f"{i}.txt"), arg) as f:
                    f.write(summary)