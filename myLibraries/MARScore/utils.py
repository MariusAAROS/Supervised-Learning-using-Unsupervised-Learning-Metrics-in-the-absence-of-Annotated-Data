from transformers import BertTokenizer, BertModel
import torch
from umap import UMAP
import plotly.graph_objects as go
import numpy as np
from matplotlib import cm
from sklearn.feature_extraction.text import TfidfVectorizer


def tokenizeCorpus(corpus, model=BertModel.from_pretrained('bert-base-uncased', 
                                                           output_hidden_states=True), 
                           tokenizer = BertTokenizer.from_pretrained('bert-base-uncased'), 
                           model_input_size=512):
    """
    Tokenize a text in order to create a dynamic embedding.

    :param1 corpus (string): Text to be tokenized.
    :param2 model (transformer): Model used to create the embedding.
    :param3 tokenizer (transformer): Tokenizer used to encode text.
    :param4 model_input_size (int): Maximum receivable input size for the transformer model.

    :output1 output (dict): Dictionnary containing the transformer model's weigths.
    :output2 labels (tensor): Text correponding to each encoded element (usefull for visualization). 
    """
    def flatten(l):
        """
        Flattens an array. Copied from stack overflow: https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists.

        :output (list): Array reduced of one dimension.
        """
        return [item for sublist in l for item in sublist]
    input_size = model_input_size - 1
    corpusWords = corpus.split(" ")
    splited = [" ".join(corpusWords[i:i+input_size]) for i in range(0, len(corpusWords), input_size)]

    input_ids = []
    attention_masks = []
    for sentence in splited:
        encoded = tokenizer.encode_plus(sentence, 
                                        add_special_tokens=True,
                                        max_length=input_size+1,
                                        padding="max_length",
                                        return_attention_mask=True,
                                        return_tensors='pt',
                                        truncation=True)
        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])

    inputs_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    temp = flatten([batch.tolist() for batch in input_ids])
    labels = np.array(temp)
    labels = labels.reshape((labels.shape[0]*labels.shape[1]))
    labels = tokenizer.convert_ids_to_tokens(labels)
    with torch.no_grad():
        output = model(inputs_ids, attention_mask=attention_masks)
    return output, labels

def vectorizeCorpus(model_output, allStates=True, tolist=True):
    """
    Transforms an encoded text to word-level vectors using a transformer model.

    :param1 model_output (dict): Dictionnary containing the transformer model's weigths.
    :param2 allStates (bool): Defines if all of the transformer's hidden states shall be used or only the last hidden state.

    :output embs (list): List of dynamics embeddings for each word of the initial corpus.
    """
    if allStates==True:
        hidden_states = model_output.hidden_states
    else:
        hidden_states = [model_output.last_hidden_state]
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = token_embeddings.permute(1,2,0,3)
    embs = []
    for batch in token_embeddings:
        for token in batch:
            emb = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
            embs.append(emb)
    if tolist:
        embs = [emb.tolist() for emb in embs]
    return embs

def tf(text):
    vectorizer = TfidfVectorizer(use_idf=False, norm=None, tokenizer=lambda x: x, lowercase=False)
    tf_values = vectorizer.fit_transform([text]).toarray()[0]
    tf_dict = {word: tf_values[index] for word, index in vectorizer.vocabulary_.items()}
    return tf_dict

def cleanVectors(data, labels):
    """
    Removes vectors associated with noisy words such as stop words, punctuation, and BERT separator tokens.

    :param1 data (list): List of words embeddings.
    :param2 labels (list): List of text token associated with each embedding.

    :output new (list): Cleansed list of words embeddings.
    """
    token_indexes = [i for i in range(len(labels)) if labels[i] != "[PAD]" and labels[i] != "[CLS]" and labels[i] != "[SEP]" and len(labels[i])>2]
    #new = []
    #for k in range(len(data)):
        #data[k] = [data[i] for i in range(len(data[k])) if i in token_indexes]
    #    if i in token_indexes:
    new = [data[i] for i in range(len(data)) if i in token_indexes]

    return new

def visualizeCorpus(embs, labels, embs_gold=None, labels_gold=None, labels_cluster=None, tf_values=None, dim=2):
    """
    Create a visual representation of the vectorized corpus using Plotly express. 

    :param1 embs (list): List of dynamics embeddings for each word of the initial corpus.
    :param2 labels (tensor): Text correponding to each encoded element.
    :param3 embs_gold (list): List of dynamics embeddings for each word of the gold reference.
    :param4 labels_gold (tensor): Text correponding to each encoded element.
    :param5 tf_values (dict): Dictionnary of Term-Frequency for each token of the text.
    :param6 dim (int): Number of dimensions wanted for the visualization (only 1 and 2 are available because they are the most usefull).
    """
    def colorize(label=None, glabels=[], clabel=None, cmap=[], mode="unclustered"):
        """
        Colorize vector's projections depending on the context. 

        :param1 mode (string): Equals to <clustered>, <unclustered> to respectively colorize depending on gold's, cluster's belonging.

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
        cmap = cm.get_cmap('viridis', len(set(labels_cluster))).colors

    if dim == 1:
        umap1D = UMAP(n_components=1, init='random', random_state=0)
        proj1D = umap1D.fit_transform(formated_embs).T

        data = {"x": proj1D[0],
                "labels": labels,
                "clusters": labels_cluster}
        
        for k in data.keys():
            data[k] = [data[k][i] for i in range(len(data[k])) if i in token_indexes]

        if comp_gold:
            token_indexes_gold = [i for i in range(len(labels_gold)) if labels_gold[i] != "[PAD]" and labels_gold[i] != "[CLS]" and labels_gold[i] != "[SEP]" and len(labels_gold[i])>2]
            proj1D_gold = umap1D.fit_transform(formated_embs_gold).T
            data_gold = {"x": proj1D_gold[0],
                        "labels": labels_gold}
            for k in data_gold.keys():
                data_gold[k] = [data_gold[k][i] for i in range(len(data_gold[k])) if i in token_indexes_gold]

        traces = []
        for i in range(len(data['x'])):
            if data["clusters"].all() != None:
                color = colorize(clables=data["clusters"][i], cmap=cmap, mode="clustered")
            else:
                color = colorize(label=data['labels'][i], glabels=data_gold['labels'], mode="unclustered")
            trace = go.Scatter(
                x=[data['x'][i]],
                mode='markers',
                marker=dict(size=9, color=color),
                line=dict(width=2, color="DarkSlateGrey"),
                text=["token: "+str(data['labels'][i]),
                      "tf   : "+str(tf_values[data['labels'][i]])],
                name=data['labels'][i]
            )
            traces.append(trace)
        if comp_gold:
            for i in range(len(data_gold['x'])):
                trace = go.Scatter(
                    x=[data_gold['x'][i]],
                    mode='markers',
                    marker=dict(size=9, color='red'),
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
        umap2D = UMAP(n_components=2, init='random', random_state=0)
        proj2D = umap2D.fit_transform(formated_embs).T

        data = {"x": proj2D[0],
                "y": proj2D[1],
                "labels": labels,
                "clusters": labels_cluster}
        
        for k in data.keys():
            data[k] = [data[k][i] for i in range(len(data[k])) if i in token_indexes]

        if comp_gold:
            token_indexes_gold = [i for i in range(len(labels_gold)) if labels_gold[i] != "[PAD]" and labels_gold[i] != "[CLS]" and labels_gold[i] != "[SEP]" and len(labels_gold[i])>2]
            proj2D_gold = umap2D.fit_transform(formated_embs_gold).T
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
                text=["token: "+str(data['labels'][i]),
                      "tf   : "+str(tf_values[data['labels'][i]])],
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

