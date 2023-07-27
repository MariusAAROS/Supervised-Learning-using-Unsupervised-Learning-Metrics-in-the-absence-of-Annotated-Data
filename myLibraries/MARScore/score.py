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
from matplotlib import pyplot as plt
import numpy as np

class MARSCore():
    def __init__(self, 
                 corpus, 
                 gold,
                 model=BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True), 
                 tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'),
                 clusterizer=hdbscan.HDBSCAN(),
                 dim_reductor=deserialize(r"C:\Pro\Stages\A4 - DVRC\Work\Ressources\umap2D.pkl"),
                 ratio=2,
                 n_allowed_elements=None,
                 printRange = range(1),
                 low_memory=False,
                 precision_level="c",
                 expe_params=None,
                 extraction_method="concat_l4",
                 lambda_param=0.5) -> None:
        """
        Constructor of the MARScore class.

        :param1 self (MARScore): MARScore Object (see __init__ function for more details).
        :param2 corpus (list): List of texts to summarize.
        :param3 gold (list): List of gold summaries.
        :param4 model (transformer): Transformer model used compute dynamic embeddings.
        :param5 tokenizer (transformer): Transformer used to create token from a plain text. 
        :param6 clusterizer (model): Model used to clusterize the dynamics embeddings.
        :param7 dim_reductor (model): Dimension reduction algorithm (UMAP as default).
        :param8 ratio (float or int): Number determining how much the reference text will be shortened.
        :param9 n_allowed_elements (int): Number of characters/sentences (depending on precision_level parameter) allowed in the summary. Overrides ratio parameter.
        :param10 printRange (range): Range of corpus that should be displayed when the Refiner object in printed.
        :param11 low_memory (bool): If set to True, stores many informations about computation allowing to compute class printing and visualization.
        :param12 precision_level (string): Defines the method used to calculate the limit length of the output summary {c: character level, s: sentence level}.
        :param13 expe_params (dict): Differents parameters usefull for experimentation purpose.
        :param14 extraction_method (str): Method of extraction for BERT embeddings.
        :param15 lambda_param (float): Value between 0 and 1 allowing to tune between relevancy and redundancy in the solver (closer to 1 is higher relevancy interest, closer to 0 means higher redundancy interest).
        """
        self.corpus = corpus
        self.gold = gold
        self.summaries = []
        self.model = model
        self.tokenizer = tokenizer
        self.clusterizer = clusterizer
        self.dim_reductor = dim_reductor
        self.extraction_method = extraction_method
        self.ratio = ratio
        self.n_allowed_elements = n_allowed_elements
        self.vectors = []
        self.reduced_vectors = []
        self.labels = []
        self.sentenced_labels = []
        self.clusters_labels = []
        self.clusters_tfs = []
        self.tokens_tfs = []
        self.similarity_matrices = []
        self.processedCorpus = []
        self.selectedIndexes = []
        self.scores = []
        self.printRange = printRange
        self.low_memory = low_memory
        self.precision_level = precision_level
        self.expe_params = expe_params
        self.written_summaries = 0
        self.lambda_param = lambda_param
    
    def compute(self, checkpoints=False, saveRate=50):
        """
        Creates extractive summaries from the corpus attribute using dynamic embedding, high dimensionnal clustering and MIP/ILP solver.

        :param1 self (MARScore): MARScore Object (see __init__ function for more details).
        :param2 checkpoints (bool): Indicates whether the refining should save partial outputs along computation to prevent from losing data in the context of a crash.
        :param3 saveRate (int): Only applicable id safe equals True. Specify the number of consicutive iterations after which a checkpoint should be created. 
        """
        if checkpoints:
            iter = 0
            start = datetime.now()
            createFolder = True

        if not(self.low_memory):
            scaler = MinMaxScaler()

        for indiv in self.corpus:
            #creation of embeddings
            o, l = tokenizeCorpus(indiv)
            v = vectorizeCorpus(o, method=self.extraction_method)
            clean_indexes = cleanAll2(l)
            
            sentenced_tokens = corpusToSentences(indiv)
            
            #sentenced_tokens = [sent.replace(".", " .") for sent in sentenced_tokens]
            #sentenced_tokens = [sent.replace("'", "' ") for sent in sentenced_tokens]
            indiv_mask = [i for i, sent in enumerate(sentenced_tokens) for _ in sent.split(" ")]
            i=0
            j=0
            sentences_mask = []
            while j < len(l):
                sentences_mask.append(indiv_mask[i])
                if "#" not in l[j]:
                    i+=1
                j+=1

            """
            i=-1
            j=0
            sentences_mask = []
            while j < len(l):
                if "#" not in l[j]:
                    i+=1
                sentences_mask.append(indiv_mask[i])
                j+=1
            """

            """
            ti = 0
            si = 0
            sentences_mask = []
            flat_sentenced_tokens = [item for row in sentenced_tokens for item in row]
            while ti < len(l) and si < len(indiv_mask)-1:
                if "##" not in l[ti]:
                    if len(l[ti]) == len(flat_sentenced_tokens[si]):
                        sentences_mask.append(indiv_mask[si])
                        si += 1
                        ti += 1
                    elif len(l[ti]) < len(flat_sentenced_tokens[si]) and l[ti] in flat_sentenced_tokens[si]:
                        count = 1
                        while l[ti + count] in flat_sentenced_tokens[si]:
                            count += 1
                            sentences_mask.append(indiv_mask[si])
                        ti += count
                        si += 1
                    else:
                        print("error")
                else:
                    sentences_mask.append(indiv_mask[si])
                    ti += 1
            """
            v = [v[i] for i in range(len(v)) if i in clean_indexes]
            l = [l[i] for i in range(len(l)) if i in clean_indexes]
            sentences_mask = [sentences_mask[i] for i in range(len(sentences_mask)) if i in clean_indexes]
            sentenced_tokens = [sentenced_tokens[i] for i in range(len(sentenced_tokens)) if i in clean_indexes]

            if not(self.low_memory):
                self.vectors.append(v)
                self.labels.append(l)
                self.sentenced_labels.append(sentenced_tokens)


            #clusterization
            if not(self.low_memory):
                try: distance_metric = self.clusterizer.metric
                except: distance_metric = "euclidean"
                distances = pairwise_distances(v, metric=distance_metric)
                normalized_distances = scaler.fit_transform(distances)
                normalized_distances = 0.5 * (normalized_distances + normalized_distances.T)
                self.similarity_matrices.append(normalized_distances)
            reduced_v, clabels = clusterizeCorpus(self.dim_reductor, v, self.clusterizer)
            if not(self.low_memory):
                self.reduced_vectors.append(reduced_v)
                self.clusters_labels.append(clabels)

            #TF calculation
            tf_values = tf(l)
            clusters_tf_values = clusters_tf(tf_values, l, clabels)
            if not(self.low_memory):
                self.tokens_tfs.append(tf_values)
                self.clusters_tfs.append(clusters_tf_values)
            
            #ILP computation
            root = get_git_root()
            dirpath = os.path.join(root, "myLibraries\MARScore_output")
            if self.expe_params == None:
                save_path_in = os.path.join(dirpath, "ilp_in.ilp")
                save_path_out = os.path.join(dirpath, "ilp_out.sol")
            elif "shuffled" in self.expe_params.keys():
                if self.expe_params["shuffled"]:
                    save_path_in = os.path.join(dirpath, "ilp_in_shuffled.ilp")
                    save_path_out = os.path.join(dirpath, "ilp_out_shuffled.sol")
                else:
                    save_path_in = os.path.join(dirpath, "ilp_in_regular.ilp")
                    save_path_out = os.path.join(dirpath, "ilp_out_regular.sol")

            _ = to_ilp_format_V3(save_path_in, reduced_v, l, clabels, sentences_mask, clusters_tf_values, self.ratio, self.precision_level, self.n_allowed_elements, self.lambda_param)
            
            os.system(f'glpsol --tmlim 100 --lp "{save_path_in}" -o "{save_path_out}"')
            selected = readILP(path=save_path_out)

            #summaries construction
            #sentences = [sentence.strip() for sentence in indiv.split(".")]
            sum_sentences = []
            selected_indexes_temp = []
            for i, value in enumerate(selected):
                if value == 1:
                    sum_sentences.append(sentenced_tokens[i])
                    selected_indexes_temp.append(i)
            self.summaries.append(" ".join(sum_sentences))
            if not(self.low_memory):
                self.selectedIndexes.append(sorted(selected_indexes_temp))
                self.processedCorpus.append(sentenced_tokens)

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

    def light_assess(self, start=0, stop=None, verbose=True):
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

            #Rouge-Score computation
            rougeScorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rougeScore = [rougeScorer.score(c, r) for c, r in zip(subset_gold, subset_summaries)]

            #Data formating
            rouge1_R = [round(t['rouge1'][0], 2) for t in rougeScore]
            rouge2_R = [round(t['rouge2'][0], 2) for t in rougeScore]
            rougeL_R = [round(t['rougeL'][0], 2) for t in rougeScore]

            dfCustom = pd.DataFrame({'R-1' : rouge1_R,
                                     'R-2' : rouge2_R,
                                     'R-L' : rougeL_R
                                    })
            #score storing
            self.scores = rouge2_R

            if verbose:
                printout = "Scores: \n"
                printout += dfCustom.to_string() + "\n\n"
                print(printout)

            return {"scores": dfCustom}

    def visualize(self, indiv=0, dim=2):
        """
        Generates a plotly graph in 1 or 2 dimensions of the clusterized tokens using UMAP.

        :param1 self (MARScore): MARScore Object (see __init__ function for more details).
        :param2 indiv (int): Document index for which representation is desired.
        :param3 dim (int): Target number of dimensions for the UMAP reduction. 
        """
        if not(self.low_memory):
            o_gold, l_gold = tokenizeCorpus(self.gold[indiv])
            v_gold = vectorizeCorpus(o_gold, method=self.extraction_method)
            v_gold, l_gold = cleanAll(v_gold, l_gold) 
            visualizeCorpus(self.dim_reductor, self.reduced_vectors[indiv], self.labels[indiv], v_gold, l_gold, self.clusters_labels[indiv], self.tokens_tfs[indiv], dim)
        else:
            print(f"\n{Fore.RED}Low memory mode activated: very likely that required attributes were not stored during computation{Style.RESET_ALL}\n\n")

    def cluster_distribution(self, indiv=0):
        """
        Creates a formated string with different colorization depending on the words' clusters.

        :param1 self (MARScore): MARScore Object (see __init__ function for more details).
        :param2 indiv (int): Document index for which representation is desired.        
        """
        if not(self.low_memory):
            output = "" 
            n_clusters = len(set(self.clusters_labels[indiv]))
            for token, c_label in zip(self.labels[indiv], self.clusters_labels[indiv]):
                color = c_label % n_clusters + 1  # Generate a color code (1 to clusters) based on the label
                colored_word = getattr(Fore, list(Fore.__dict__.keys())[color]) + token + Fore.RESET
                output += colored_word + ' '
            output = output.replace(" ##", "")
            output = output.replace(".", ".\n")
            output = output.strip()
            return output
        else:
            print(f"\n{Fore.RED}Low memory mode activated: very likely that required attributes were not stored during computation{Style.RESET_ALL}\n\n")
            return -1

    def save(self, runtime=None, new=True, pace=50):
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
                main_folder_path = os.path.join(get_git_root(), r"myLibraries\MARScore_output\results")
            elif "shuffled" in self.expe_params.keys():
                if self.expe_params["shuffled"]:
                    main_folder_path = os.path.join(get_git_root(), r"myLibraries\MARScore_output\shuffled")
                else:
                    main_folder_path = os.path.join(get_git_root(), r"myLibraries\MARScore_output\regular")
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
            pos = len(self.summaries)-self.written_summaries

            for i, summary in enumerate(self.summaries[len(self.summaries)-pos:]):
                if new:
                    arg = "w"
                else:
                    arg = "a"
                with open(os.path.join(sum_path, f"{i+self.written_summaries}.txt"), arg) as f:
                    f.write(summary)

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