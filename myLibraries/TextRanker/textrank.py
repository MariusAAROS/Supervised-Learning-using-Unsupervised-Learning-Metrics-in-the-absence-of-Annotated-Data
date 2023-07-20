import spacy
import pytextrank
from operator import itemgetter
from math import sqrt

def text_rank(corpus, limit_phrases=4, limit_sentences=2):
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank", last=True)
    summaries = []

    for text in corpus:
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
        summaries.append(result[:-1])
    return summaries