from transformers import BertTokenizer, BertModel
import torch
from umap import UMAP
import plotly.graph_objects as go
import numpy as np


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

def vectorizeCorpus(model_output, allStates=True):
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
    return embs

def visualizeCorpus(embs, labels, embs_gold=None, labels_gold=None, dim=2):
    """
    Create a visual representation of the vectorized corpus using Plotly express. 

    :param1 embs (list): List of dynamics embeddings for each word of the initial corpus.
    :param2 labels (tensor): Text correponding to each encoded element.
    :param3 embs_gold (list): List of dynamics embeddings for each word of the gold reference.
    :param4 labels_gold (tensor): Text correponding to each encoded element.
    :param5 dim (int): Number of dimensions wanted for the visualization (only 1 and 2 are available because they are the most usefull).
    """
    comp_gold = True if embs_gold != None and labels_gold != None else False

    formated_embs = [token.tolist() for token in embs]
    formated_embs = np.array(formated_embs)
    formated_embs_gold = [token.tolist() for token in embs_gold]
    formated_embs_gold = np.array(formated_embs_gold)
    token_indexes = [i for i in range(len(labels)) if labels[i] != "[PAD]" and labels[i] != "[CLS]" and labels[i] != "[SEP]" and len(labels[i])>2]

    if dim == 1:
        umap1D = UMAP(n_components=1, init='random', random_state=0)
        proj1D = umap1D.fit_transform(formated_embs).T

        data = {"x": proj1D[0],
                "labels": labels}
        
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
            if comp_gold:
                color = 'green' if data["labels"][i] in data_gold["labels"] else 'red'
            else:
                color = 'red'
            trace = go.Scatter(
                x=[data['x'][i]],
                mode='markers',
                marker=dict(size=6, color=color),
                text=[data['labels'][i]],
                name=data['labels'][i]
            )
            traces.append(trace)
        if comp_gold:
            for i in range(len(data_gold['x'])):
                trace = go.Scatter(
                    x=[data_gold['x'][i]],
                    mode='markers',
                    marker=dict(size=6, color='gold'),
                    text=[data_gold['labels'][i]],
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
                "labels": labels}
        
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
            if comp_gold:
                color = 'green' if data["labels"][i] in data_gold["labels"] else 'red'
            else:
                color = 'red'
            trace = go.Scatter(
                x=[data['x'][i]],
                y=[data['y'][i]],
                mode='markers',
                marker=dict(size=6, color=color),
                text=[data['labels'][i]],
                name=data['labels'][i]
            )
            traces.append(trace)
        if comp_gold:
            for i in range(len(data_gold['x'])):
                trace = go.Scatter(
                    x=[data_gold['x'][i]],
                    y=[data_gold['y'][i]],
                    mode='markers',
                    marker=dict(size=6, color='gold'),
                    text=[data_gold['labels'][i]],
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