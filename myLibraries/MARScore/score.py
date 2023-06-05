from MARScore.utils import *
import hdbscan
import os

class MARSCore():
    def __init__(self, corpus, model=BertModel.from_pretrained('bert-base-uncased', 
                                                               output_hidden_states=True), 
                               tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')) -> None:
        """
        Constructor of the MARScore class.

        :param1 corpus (list): List of texts to summarize.
        :param2 model (transformer): Transformer model used compute dynamic embeddings.
        :param3 tokenizer (transformer) Transformer used to create token from a plain text. 
        """
        self.corpus = corpus
        self.summary = []
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

            #summary construction
            sentences = indiv.split(".")
            sentences.pop()
            sum_sentences = []
            for i, value in enumerate(selected):
                if value == 1:
                    sum_sentences.append(sentences[i]+".")
            self.summary.append(" ".join(sum_sentences))