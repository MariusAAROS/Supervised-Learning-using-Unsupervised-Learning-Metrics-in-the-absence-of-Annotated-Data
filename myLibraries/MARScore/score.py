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
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

class MARSCore():
    def __init__(self, 
                 corpus, 
                 gold,
                 model=BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True), 
                 tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'),
                 clusterizer=hdbscan.HDBSCAN(),
                 ratio=2,
                 printRange = range(1),
                 compute_similarity=True) -> None:
        """
        Constructor of the MARScore class.

        :param1 self (MARScore): MARScore Object (see __init__ function for more details).
        :param2 corpus (list): List of texts to summarize.
        :param3 gold (list): List of gold summaries.
        :param4 model (transformer): Transformer model used compute dynamic embeddings.
        :param5 tokenizer (transformer): Transformer used to create token from a plain text. 
        :param6 clusterizer (model): Model used to clusterize the dynamics embeddings.
        :param7 ratio (float or int): Number determining how much the reference text will be shortened.
        :param8 printRange (range): Range of corpus that should be displayed when the Refiner object in printed.
        """
        self.corpus = corpus
        self.gold = gold
        self.summaries = []
        self.model = model
        self.tokenizer = tokenizer
        self.clusterizer = clusterizer
        self.ratio = ratio
        self.vectors = []
        self.labels = []
        self.clusters_labels = []
        self.clusters_tfs = []
        self.similarity_matrices = []
        self.processedCorpus = []
        self.selectedIndexes = []
        self.scores = []
        self.printRange = printRange
        self.compute_similarity = compute_similarity
    
    def compute(self, checkpoints=False, saveRate=50):
        """
        Creates extractive summaries from the corpus attribute using dynamic embedding, high dimensionnal clustering and MIP/ILP solver.

        :param1 self (MARScore): MARScore Object (see __init__ function for more details).
        """
        if checkpoints:
            iter = 0
            start = datetime.now()
            createFolder = True

        for indiv in self.corpus:
            #creation of embeddings
            o, l = tokenizeCorpus(indiv)
            v = vectorizeCorpus(o)
            v, l = cleanAll(v, l)
            self.vectors.append(v)
            self.labels.append(l)

            #clusterization
            if self.compute_similarity:
                try: distance_metric = self.clusterizer.metric
                except: distance_metric = "euclidean"
                distances = pairwise_distances(v, metric=distance_metric)
                scaler = MinMaxScaler()
                normalized_distances = scaler.fit_transform(distances)
                self.similarity_matrices.append(normalized_distances)
            clabels = clusterizeCorpus(v, self.clusterizer)
            self.clusters_labels.append(clabels)

            #TF calculation
            tf_values = tf(l)
            clusters_tf_values = clusters_tf(tf_values, l, clabels)
            self.clusters_tfs.append(clusters_tf_values)
            
            #ILP computation
            _ = to_ilp_format(l, clabels, clusters_tf_values, self.ratio)
            root = get_git_root()
            dirpath = os.path.join(root, "myLibraries\MARScore_output")
            os.system(f'glpsol --tmlim 100 --lp "{os.path.join(dirpath, "ilp_in.ilp")}" -o "{os.path.join(dirpath, "ilp_out.sol")}"')
            selected = readILP()

            #summaries construction
            sentences = [sentence.strip() for sentence in indiv.split(".")]
            sentences.pop()
            sum_sentences = []
            selected_indexes_temp = []
            for i, value in enumerate(selected):
                if value == 1:
                    sum_sentences.append(sentences[i]+".")
                    selected_indexes_temp.append(i)
            self.selectedIndexes.append(sorted(selected_indexes_temp))
            self.summaries.append(" ".join(sum_sentences))
            self.processedCorpus.append(sentences)

            #checkpoint verification
            if checkpoints:
                if iter % saveRate == 0 and iter != 0:
                    stop = datetime.now()
                    partial_runtime = stop - start
                    self.save(runtime=partial_runtime, new=createFolder)
                    createFolder = False
                iter += 1
                
        if checkpoints:
            stop = datetime.now()
            runtime = stop - start
            self.save(runtime=runtime, new=createFolder)

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
            Saves Refiner output to a local folder.

            :param1 self (Refiner): Refiner Object (see __init__ function for more details).
            :param2 new (bool): Indicates if a new folder should be created. If false, output is append to the most recent ouput folder.
            """
            #evaluation
            start = 0
            stop = len(self.summaries)
            assessement = self.assess(start=start, stop=stop)

            #mainDf = self.to_dataframe()
            scoreDf = assessement["scores"]
            corDf = assessement["correlations"]

            #write output
            main_folder_path = os.path.join(get_git_root(), r"myLibraries\MARScore_output\results")
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
        Summarizes MARScore object to a string.

        :param1 self (MARScore): MARScore Object (see __init__ function for more details).

        :output printout (string): Summarized informations about the MARScore object.
        """

        printout = "--------MARScore OBJECT--------\n\n"
        printout += "Number of Documents : " + str(len(self.corpus)) + "\n"
        printout += "Corpus Avg Size     : " + str(int(np.average([len(x) for x in self.corpus]))+1) + "\n"
        printout += "Refined Avg Size    : " + str(int(np.average([len(x) for x in self.summaries]))+1) + "\n"

        printout += "\n-------------------------------\n"

        self.printRange = self.printRange if self.printRange.start >= 0 and self.printRange.stop < len(self.processedCorpus) else range(0, len(self.processedCorpus))

        for index in self.printRange:
            printout += f"\nCorpus no.{index+1} : {str(self.scores[index]*100)+'%' if self.scores != [] and self.scores != -1 else ''}\n" + str(".\n".join([f"{Fore.LIGHTGREEN_EX}{self.processedCorpus[index][i]}{Style.RESET_ALL}"
                                                        if i in self.selectedIndexes[index]
                                                        else f"{Fore.RED}{self.processedCorpus[index][i]}{Style.RESET_ALL}"
                                                        for i in range(len(self.processedCorpus[index]))])) + "." + "\n"
        
        printout += "\n-------------------------------"
        return printout