from nltk.stem import PorterStemmer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def issymbol(character):
    ascii = ord(character)
    non_symbols = list(range(48,58)) + list(range(65,91)) + list(range(97,123))
    if ascii not in non_symbols:
        return True
    else:
        return False

def findshape(word):
    if word[0].isupper():
        return 'capitalize'
    elif word[0].islower():
        return 'lowercase'
    elif len(word)==1 and issymbol(word):
        return 'punct'
    elif word.isdigit():
        return 'number'
    elif word.startswith('__') and word.endswith('__'):
        return 'wildcard'
    else:
        return 'other'

def encode(tags, stems):
    assert len(tags)==len(stems), "Error. Length of stems and tags not same"
    l = len(tags)
    train_X = []
    for i in range(len(tags)):
        x = {
        'pos': tags[i][1],
        'lemma': stems[i],
        'shape': findshape(tags[i][0])
        }

        if i<l-1:
            x['next-pos'] = tags[i+1][1]
            x['next-lemma'] = stems[i+1]
            x['next-shape'] = findshape(tags[i+1][0])
            #x['next-word'] = tags[i+1][0]
        if i<l-2:
            x['next-next-pos'] = tags[i+2][1]
            x['next-next-lemma'] = stems[i+2]
            x['next-next-shape'] = findshape(tags[i+2][0])
            #x['next-next-word'] = tags[i+2][0]
        if i>0:
            x['prev-pos'] = tags[i-1][1]
            x['prev-lemma'] = stems[i-1]
            x['prev-shape'] = findshape(tags[i-1][0])
            #x['prev-word'] = tags[i-1][0]
        if i>1:
            x['prev-prev-pos'] = tags[i-2][1]
            x['prev-prevplemma'] = stems[i-2]
            x['prev-prev-shape'] = findshape(tags[i-2][0])
            #x['prev-prev-word'] = tags[i-2][0]

        train_X.append(x)

    return train_X
