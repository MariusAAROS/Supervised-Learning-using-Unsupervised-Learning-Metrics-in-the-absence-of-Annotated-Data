from transformers import BertTokenizer, BertModel
import torch
from umap import UMAP
import plotly.graph_objects as go
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from custom_score.utils import cleanString, get_git_root
import os
import re
import io
import sys
import git
from contextlib import contextmanager
from matplotlib import colormaps
import random


def tokenizeCorpus(corpus, model=BertModel.from_pretrained('bert-base-uncased', 
                                                           output_hidden_states=True), 
                           tokenizer = BertTokenizer.from_pretrained('bert-base-uncased'), 
                           model_input_size=512,
                           padding_id=0):
    """
    Tokenize a text in order to create a dynamic embedding.

    :param1 corpus (string): Text to be tokenized.
    :param2 model (transformer): Model used to create the embedding.
    :param3 tokenizer (transformer): Tokenizer used to encode text.
    :param4 model_input_size (int): Maximum receivable input size for the transformer model.
    :param5 padding_id (int): Number representing the padding token for the transfomer model in use.

    :output1 output (dict): Dictionnary containing the transformer model's weigths.
    :output2 labels (tensor): Text correponding to each encoded element (usefull for visualization). 
    """
    
    input_size = model_input_size - 15
    corpusWords = corpus.split(" ")
    splited = [" ".join(corpusWords[i:i+input_size]) for i in range(0, len(corpusWords), input_size)]
   
    b_encoded = tokenizer.batch_encode_plus(splited,
                                      add_special_tokens=True,
                                      max_length=None,
                                      padding=True,
                                      return_attention_mask=True,
                                      return_tensors='pt',
                                      truncation=False)

    input_ids = b_encoded["input_ids"]
    attention_masks = b_encoded["attention_mask"]

    input_ids = torch.flatten(input_ids)
    input_ids = input_ids[input_ids != padding_id]
    attention_masks = attention_masks[attention_masks != 0]
    padding_length = model_input_size - (input_ids.size(0) % model_input_size)
    padding_tensor = torch.full((padding_length,), padding_id)
    input_ids = torch.cat((input_ids, padding_tensor), dim=0)
    attention_masks = torch.cat((attention_masks, padding_tensor), dim=0)
    labels = tokenizer.convert_ids_to_tokens(input_ids)
    input_ids = input_ids.reshape((-1, model_input_size))
    attention_masks = attention_masks.reshape((-1, model_input_size))
    
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_masks)
    return output, labels

def vectorizeCorpus(model_output, allStates=True, tolist=True, method="concat_l4"):
    """
    Transforms an encoded text to word-level vectors using a transformer model.

    :param1 model_output (dict): Dictionnary containing the transformer model's weigths.
    :param2 allStates (bool): Defines if all of the transformer's hidden states shall be used or only the last hidden state.
    :param3 tolist (bool): If True, embeddings are converted from tensor to list before return.
    :param4 method (str): Defines the method of extraction for the word embeddings. Available method from best to worst {"concat_l4": "concatenate last hidden 4 layers", 
                                                                                                                         "sum_l4": "sum last 4 hidden layers", 
                                                                                                                         "secondToLast": "take second to last hidden layer",
                                                                                                                         "sum_all": "sum all hidden layers",
                                                                                                                         "last": "take last hidden layer",
                                                                                                                         "first": "take first hidden layer"}
    :output embs (list): List of dynamics embeddings for each word of the initial corpus.
    """
    def switch_method(method, token):
        """
        Helper function to select embedding extraction method.

        :param1 method (str): Switch parameter for method selection.
        :param2 token (Tensor): BERT layers.
        """
        if method == "concat_l4":
            return torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
        elif method == "sum_l4":
            return torch.sum(token[-4:], dim=0)
        elif method == "secondToLast":
            return token[-2]
        elif method == "sum_all":
            return torch.sum(token, dim=0)
        elif method == "last":
            return token[-1]
        elif method =="first":
            return token[0]
        else:
            raise ValueError("Invalid embedding extraction method")
        
    if allStates==True:
        hidden_states = model_output.hidden_states
    else:
        hidden_states = [model_output.last_hidden_state]
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = token_embeddings.permute(1,2,0,3)
    embs = []
    for batch in token_embeddings:
        for token in batch:
            emb = switch_method(method, token)
            embs.append(emb)
    if tolist:
        embs = [emb.tolist() for emb in embs]
    return embs

def clusterizeCorpus(reductor, embs, clusterizer):
    """"
    Clusterizes high dimensionnal vectors by reducing them first.

    :param1 reductor (model): Usually a UMAP instance.
    :param2 embs (list): High dimensionnal vectors.
    :param3 clusterizer (model): Clusterizer model, usually HDBScan or MinCut.

    :output1 proj (list): List of vectors with reduced dimensions. 
    :output2 clabels (list): List of clusters labels for each vector.
    """
    if clusterizer.__module__ == "hdbscan.hdbscan_" or ".".join(clusterizer.__module__.split(".")[:2]) == "sklearn.cluster":
        proj = reductor.fit_transform(embs)
        clusterizer.fit(proj)
        clabels = clusterizer.labels_.astype(int)
    else:
        print("\nERROR: Clusterizer not supported yet.\n")
    return proj, clabels

def tf(text):
    """
    Computes individual TFs of tokens.

    :param1 text (list): List of tokens.

    :output tf_dict (dict): Dictionnary of tokens with their corresponding IDF.
    """
    vectorizer = TfidfVectorizer(use_idf=False, norm=None, tokenizer=lambda x: x, lowercase=False, token_pattern=None)
    tf_values = vectorizer.fit_transform([text]).toarray()[0]
    tf_dict = {word: tf_values[index] for word, index in vectorizer.vocabulary_.items()}
    return tf_dict

def clusters_tf(tf_values, labels, clabels):
    """
    Computes clusters TFs from tokens individuals TFs.

    :param1 tf_value (dict): Dictionnary of tokens with their corresponding TF value.
    :param2 labels (list): List of the tokens.
    :param3 clabels (list): List of the token's clusters.

    :output clusters_tf_values (dict): Dictionnary of cumulated TF scores for each cluster. 
    """
    clusters_tf_values = {}
    for label, clabel in zip(labels, clabels):
        if clabel in clusters_tf_values.keys():
            clusters_tf_values[clabel] += tf_values[label]
        else:
            clusters_tf_values[clabel] = tf_values[label]
    return clusters_tf_values

def cleanAll(embs, labels, mode="all", ignore=["."]):
    """
    Removes vectors associated with noisy words such as stop words, punctuation, and BERT separator tokens.

    :param1 embs (list): List of words embeddings.
    :param2 labels (list): List of text token associated with each embedding.

    :output1 new_embs (list): Cleansed list of words embeddings.
    :output2 new_labels (list): Cleansed list of tokens.
    :output3 -1 (int): Error output.
    """
    token_indexes = [i for i in range(len(labels)) if (labels[i] != "[PAD]" and labels[i] != "[CLS]" and labels[i] != "[SEP]" and len(labels[i])>2) or labels[i] in ignore]
    new_embs = [embs[i] for i in range(len(embs)) if i in token_indexes]
    new_labels = [labels[i] for i in range(len(labels)) if i in token_indexes]


    if mode == "all":
        return new_embs, new_labels
    elif mode == "emb":
        return new_embs
    elif mode == "lab":
        return new_labels
    else:
        return -1

def visualizeCorpus(reductor, embs, labels, embs_gold=None, labels_gold=None, labels_cluster=None, tf_values=None, dim=2):
    """
    Create a visual representation of the vectorized corpus using Plotly express. 

    :param1 reductor (model): Dimension reduction algorithm (UMAP as default).
    :param1 embs (list): List of dynamics embeddings for each word of the initial corpus.
    :param2 labels (tensor): Text correponding to each encoded element.
    :param3 embs_gold (list): List of dynamics embeddings for each word of the gold reference.
    :param4 labels_gold (tensor): Text correponding to each encoded element.
    :param5 labels_cluster (list): List of the token's clusters.
    :param5 tf_values (dict): Dictionnary of Term-Frequency for each token of the text.
    :param6 dim (int): Number of dimensions wanted for the visualization (only 1 and 2 are available because they are the most usefull).
    """
    def colorize(label=None, glabels=[], clabel=None, cmap=[], mode="unclustered"):
        """
        Colorize vector's projections depending on the context. 

        :param1 label (string): Single Token.
        :param2 glabel (list): List of gold tokens.
        :param3 clabel (int): Cluster's index of the current token.
        :param4 cmap (color_map): Matplotlib color map.
        :param5 mode (string): Equals to <clustered>, <unclustered> to respectively colorize depending on gold's, cluster's belonging.

        :output color (string): CSS text color.  
        """
        comp_gold = True if label != None and glabels != [] else False
        assert label != None or glabels != [] or clabel != None, "ERROR: No labels detected"
        if mode == "unclustered":
            if comp_gold:
                color = "green" if label in glabels else 'red'
            else:
                color = "red"
        elif mode == "clustered":
            if clabel != None and cmap != []:
                color = cmap[clabel]
            else:
                color = "black"
        return color
    def create_word_dictionary(words):
        """
        Transforms list of words to dictionnary of words with value 1 (used to create TF default value).

        :param1 words (list): List of words.

        :output word_dict (dict): Dictionnary of words.
        """
        word_dict = {}
        for word in words:
            if word not in word_dict:
                word_dict[word] = 1
        return word_dict
    
    if tf_values == None:
        tf_values = create_word_dictionary(labels)

    comp_gold = True if embs_gold != None and labels_gold != None else False
    
    formated_embs = np.array(embs)
    formated_embs_gold = np.array(embs_gold)

    token_indexes = [i for i in range(len(labels)) if labels[i] != "[PAD]" and labels[i] != "[CLS]" and labels[i] != "[SEP]" and len(labels[i])>2]

    if labels_cluster.all() != None:
        cmap = colormaps["viridis"].colors

    if dim == 1:
        if len(embs[0]) != 1:
            umap1D = UMAP(n_components=1, init='random', random_state=0)
            proj1D = umap1D.fit_transform(formated_embs).T
        else:
            proj1D = np.transpose(embs)

        data = {"x": proj1D[0],
                "labels": labels,
                "clusters": labels_cluster}
        
        for k in data.keys():
            data[k] = [data[k][i] for i in range(len(data[k])) if i in token_indexes]

        if comp_gold:
            token_indexes_gold = [i for i in range(len(labels_gold)) if labels_gold[i] != "[PAD]" and labels_gold[i] != "[CLS]" and labels_gold[i] != "[SEP]" and len(labels_gold[i])>2]
            if len(embs[0]) != 1:
                formated_embs_gold = reductor.transform(formated_embs_gold)
                proj1D_gold = umap1D.transform(formated_embs_gold).T
            else:
                proj1D_gold = reductor.transform(formated_embs_gold).T
            data_gold = {"x": proj1D_gold[0],
                        "labels": labels_gold}
            for k in data_gold.keys():
                data_gold[k] = [data_gold[k][i] for i in range(len(data_gold[k])) if i in token_indexes_gold]

        traces = []
        for i in range(len(data['x'])):
            if data["clusters"] != None:
                color = colorize(clabel=data["clusters"][i], cmap=cmap, mode="clustered")
            else:
                color = colorize(label=data['labels'][i], glabels=data_gold['labels'], mode="unclustered")
            trace = go.Scatter(
                x=[data['x'][i]],
                mode='markers',
                marker=dict(size=9, color=color),
                line=dict(width=2, color="DarkSlateGrey"),
                text=["token: "+str(data['labels'][i])+" || "+"tf   : "+str(tf_values[data['labels'][i]])],
                name=data['labels'][i]
            )
            traces.append(trace)
        if comp_gold:
            for i in range(len(data_gold['x'])):
                trace = go.Scatter(
                    x=[data_gold['x'][i]],
                    mode='markers',
                    marker=dict(size=9, color='red'),
                    marker_symbol="diamond",
                    line=dict(width=2, color="DarkSlateGrey"),
                    text=["token: "+str(data_gold['labels'][i])],
                    name=data_gold['labels'][i]
                )
                traces.append(trace)

        layout = go.Layout(
            title='1D Scatter Plot',
            scene=dict(
                xaxis=dict(title='X')
            )
        )
        fig = go.Figure(data=traces, layout=layout)
        fig.show()

    elif dim == 2:
        if len(embs[0]) != 2:
            umap2D = UMAP(n_components=2, init='random', random_state=0)
            proj2D = umap2D.fit_transform(formated_embs).T
        else:
            proj2D = np.transpose(embs)

        data = {"x": proj2D[0],
                "y": proj2D[1],
                "labels": labels,
                "clusters": labels_cluster}
        
        for k in data.keys():
            data[k] = [data[k][i] for i in range(len(data[k])) if i in token_indexes]

        if comp_gold:
            token_indexes_gold = [i for i in range(len(labels_gold)) if labels_gold[i] != "[PAD]" and labels_gold[i] != "[CLS]" and labels_gold[i] != "[SEP]" and len(labels_gold[i])>2]
            if len(embs[0]) != 2:
                formated_embs_gold = reductor.transform(formated_embs_gold)
                proj2D_gold = umap2D.transform(formated_embs_gold).T
            else:
                proj2D_gold = reductor.transform(formated_embs_gold).T
            data_gold = {"x": proj2D_gold[0],
                         "y": proj2D_gold[1],
                         "labels": labels_gold}
            for k in data_gold.keys():
                data_gold[k] = [data_gold[k][i] for i in range(len(data_gold[k])) if i in token_indexes_gold]

        traces = []
        for i in range(len(data['x'])):
            if data["clusters"] != None:
                color = colorize(clabel=data["clusters"][i], cmap=cmap, mode="clustered")
            else:
                color = colorize(labels=data['labels'][i], glabels=data_gold['labels'], mode="unclustered")
            trace = go.Scatter(
                x=[data['x'][i]],
                y=[data['y'][i]],
                mode='markers',
                marker=dict(size=9, color=color),
                line=dict(width=2, color="DarkSlateGrey"),
                text=["token: "+str(data['labels'][i]) +" || "+"tf   : "+str(tf_values[data['labels'][i]])+" || cluster: "+str(data["clusters"][i])],
                name=data['labels'][i]
            )
            traces.append(trace)
        if comp_gold:
            for i in range(len(data_gold['x'])):
                trace = go.Scatter(
                    x=[data_gold['x'][i]],
                    y=[data_gold['y'][i]],
                    mode='markers',
                    marker=dict(size=9, color='red'),
                    marker_symbol="diamond",
                    line=dict(width=2, color="DarkSlateGrey"),
                    text=["token: "+str(data_gold['labels'][i])],
                    name=data_gold['labels'][i]
                )
                traces.append(trace)

        layout = go.Layout(
            title='2D Scatter Plot',
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y')
            )
        )
        fig = go.Figure(data=traces, layout=layout)
        fig.show()

def to_ilp_format(path, labels, clabels, clusters_tf_values, ratio, precision_level, n_allowed_elements, save=True, verbose=False):
    """
    Transforms a text to an ILP model.

    :param1 path (string): Absolute path where ILP output should be save. 
    :param2 labels (list): List of text token associated with each embedding.
    :param3 clabels (list): List of token's cluster index.
    :param4 clusters_tf_values (dict): Dictionnary of cumulated TF scores for each cluster.
    :param5 save (bool): Save the output to file if set to True.
    :param6 verbose (bool): Add verbose if set to True.

    :output output (string): Text formatted to with respect to ILP's requirements.
    """
    def scale_dict(d):
        """
        MinMax scales values of a dictionnary.

        :param1 d (dict): Dictionnary to scale.

        :output norm_d (dict): Dictionnary with normalized values.
        """
        values = d.values()
        min_ = min(values)
        max_ = max(values)
        norm_d = {key: ((v - min_ ) / (max_ - min_) )  for (key, v) in d.items()}
        return norm_d
    
    #create sentence dictionnary for clusters
    sentence_index = 0
    sentences_map = {0: []}
    nb_sentences = labels.count(".") if labels[-1] == "." else labels.count(".")+1
    for cluster_index, token in zip(clabels, labels):
        if cluster_index in sentences_map.keys():
            sentences_map[cluster_index].append(sentence_index)
        else:
            sentences_map[cluster_index] = [sentence_index]
        if token == ".":
            sentence_index += 1
    try:
        sentences_map.pop(-1)
    except KeyError:
        pass

    #create sentence count dictionnary for clusters
    sentences_map_count = {}
    for cluster_index in sentences_map.keys():
        sentences_map_count[cluster_index] = {}
        for sentence_index in sentences_map[cluster_index]:
            sentences_map_count[cluster_index][sentence_index] = list(sentences_map[cluster_index]).count(sentence_index)

    #transform sentence_map to set
    for k in sentences_map.keys():
        sentences_map[k] = set(sentences_map[k])

    
    #create sentence dictionnary for length
    sentence_index = 0
    sentences_lens = [0]*(nb_sentences)
    for token in labels:
        sentences_lens[sentence_index] += len(token)
        if token == ".":
            sentence_index += 1
    if precision_level == "c":
        total_len = np.sum(sentences_lens)
    elif precision_level == "s":
        total_len = nb_sentences
    if n_allowed_elements == None:
        target_len = int(total_len/ratio)
    else:
        target_len = int(n_allowed_elements)

    #define scoring function
    try:
        clusters_tf_values.pop(-1)
    except KeyError:
        pass
    #norm_clusters_tf_values = scale_dict(clusters_tf_values)
    output = "Maximize\nscore:"
    for i, k in enumerate(sorted(clusters_tf_values.keys())):
        if int(clusters_tf_values[k]) < 0:
            output += f" - {-int(clusters_tf_values[k])} c{i}"
        else:
            output += f" + {int(clusters_tf_values[k])} c{i}"
    scaler = MinMaxScaler()
    norm_sentences_lens = list(scaler.fit_transform(np.array(sentences_lens).reshape(-1, 1)).reshape(-1))
    for i, length in enumerate(norm_sentences_lens):
        output += f" - {round(length, 3)} s{i}"
         
    #define constraints
    output += "\n\nSubject To\n"
    for i, k in enumerate(sorted(sentences_map.keys())):
        output += f"index_{i}:"
        for sentence_index in sorted(sentences_map[k]):
            output += f" {sentences_map_count[k][sentence_index]} s{sentence_index} +"
        #output = output[:-2] + f" - {clusters_tf_values[k]} c{i} >= 0" + "\n"
        output = output[:-2] + f" - c{i} >= 0" + "\n"

    #define sentence length
    len_template = ""
    if precision_level == "c":
        for i in range(nb_sentences):
            len_template += f" {int(sentences_lens[i])} s{i} +"
        len_template = len_template[:-2]
        output += "length:" + len_template + f" <= {target_len}" + "\n"
        #output += "length_min:" + len_template + f" >= {int(0.75*target_len)}" + "\n"
    elif precision_level == "s":
        for i in range(nb_sentences):
            len_template += f" s{i} +"
        len_template = len_template[:-2]
        output += "length:" + len_template + f" <= {target_len}" + "\n"

    #declare cluster variables
    output += "\n\n\nBinary\n"
    for i in range(len(clusters_tf_values.keys())):
        output += f"c{i}\n"

    #declare sentences variables
    for i in range(nb_sentences):
        output += f"s{i}\n"
    output = output[:-1]

    #end file
    output += "\nEnd"

    if save:
        with open(path, "w") as text_file:
            text_file.write(output)
            text_file.close()
        if verbose:
            print("\nSave successful")

    return output

def readILP(rel_path="myLibraries\MARScore_output\ilp_out.sol", path=None):
    """
    Reads and parses the output value of an ILP model.

    :param1 rel_path (string): Relative path to the desired ILP output file.
    :param2 path (string): Absolute path to replace relative path. 

    :output result (list): List of sentence selection performed by the ILP model.
    """
    if path == None:
        root = get_git_root()
        path = os.path.join(root, rel_path)
    with open(path, "r") as f:
        lines = f.readlines()
        f.close()
    
    sentences_lines = [line for line in lines if re.search(r"s\d", line)]

    sorted_lines = sorted(sentences_lines, key=lambda line: int(line.split()[1][1:]))
    result = [int(sorted_line.split()[3]) for sorted_line in sorted_lines]
    return result

def get_git_root():
    """
    Return the path to the current git directory's root.

    output git_root (str): Path of the current git directory's root.
    """
    git_repo = git.Repo(os.path.abspath(__file__), search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root

@contextmanager
def nostd():
    """
    Decorator supressing console output. Source : https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto
    """
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    sys.stdout = io.BytesIO()
    sys.stderr = io.BytesIO()
    yield
    sys.stdout = save_stdout
    sys.stderr = save_stderr

def updateFileCount(path):
    """
    :param1 path (string): Path to the file counting the number of experimentation output I already generated.

    :output updated (int): New number of output file created after the current experimentation.
    """
    if os.path.exists(path):
        with open(path, "r") as f:
            value = int(f.read())
        updated = value + 1
    else:
        updated = 1
    with open(path, "w") as f:
        f.write(str(updated))    
    return updated

def readFileCount(path):
    """
    :param1 path (string): Path to the file counting the number of experimentation output I already generated.

    :output updated (int): New number of output file created after the current experimentation.
    """
    if os.path.exists(path):
        with open(path, "r") as f:
            value = int(f.read())
    return value