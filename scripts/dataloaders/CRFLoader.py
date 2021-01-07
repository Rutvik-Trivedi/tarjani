from .BaseLoader import BaseLoader
import json
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from itertools import chain

class CRFLoader(BaseLoader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requires_save = False

    def __call__(self, intent_name, **kwargs):
        train_X = []
        train_y = []
        intent_folder='../intents'
        filepath = intent_folder+'/'+intent_name+'/intent.tarjani'
        with open(filepath, 'r') as f:
            data = json.load(f)
        for i in range(len(data['query'])):
            tokens = word_tokenize(data['query'][i])
            tags = pos_tag(tokens)
            stems = [self.lemmatizer.lemmatize(i) for i in tokens]
            x = self.encode(tags, stems)
            y = ['O']*len(x)
            for key in data['entity'][i].keys():
                if str(type(data['entity'][i][key]))=="<class 'int'>":
                    y[data['entity'][i][key]] = key
                elif not data['entity'][i][key]:
                    continue
                else:
                    for j in data['entity'][i][key]:
                        if y[j] != "O":
                            raise EntityOverwriteError("The word {} is already assigned an entity. Entity overwriting is not allowed. Please create the intent afresh".format(tags[0][j]))
                        y[j] = key
            train_X.append(x)
            train_y.append(y)

        self.labels = list(self.get_labels(train_y))

        return train_X, train_y

    def name(self):
        return 'crf_loader'

    def get_labels(self, train_y):
        return set(chain.from_iterable(train_y))

    def issymbol(self, character):
        ascii = ord(character)
        non_symbols = list(range(48,58)) + list(range(65,91)) + list(range(97,123))
        if ascii not in non_symbols:
            return True
        else:
            return False

    def findshape(self, word):
        if word[0].isupper():
            return 'capitalize'
        elif word[0].islower():
            return 'lowercase'
        elif len(word)==1 and self.issymbol(word):
            return 'punct'
        elif word.isdigit():
            return 'number'
        elif word.startswith('__') and word.endswith('__'):
            return 'wildcard'
        else:
            return 'other'

    def encode(self, tags, stems):
        assert len(tags)==len(stems), "Error. Length of stems and tags not same"
        l = len(tags)
        train_X = []
        for i in range(len(tags)):
            x = {
            'pos': tags[i][1],
            'lemma': stems[i],
            'shape': self.findshape(tags[i][0])
            }

            if i<l-1:
                x['next-pos'] = tags[i+1][1]
                x['next-lemma'] = stems[i+1]
                x['next-shape'] = self.findshape(tags[i+1][0])
                x['next-word'] = tags[i+1][0]
            if i<l-2:
                x['next-next-pos'] = tags[i+2][1]
                x['next-next-lemma'] = stems[i+2]
                x['next-next-shape'] = self.findshape(tags[i+2][0])
                x['next-next-word'] = tags[i+2][0]
            if i>0:
                x['prev-pos'] = tags[i-1][1]
                x['prev-lemma'] = stems[i-1]
                x['prev-shape'] = self.findshape(tags[i-1][0])
                x['prev-word'] = tags[i-1][0]
            if i>1:
                x['prev-prev-pos'] = tags[i-2][1]
                x['prev-prevplemma'] = stems[i-2]
                x['prev-prev-shape'] = self.findshape(tags[i-2][0])
                x['prev-prev-word'] = tags[i-2][0]

            train_X.append(x)

        return train_X

    def prepare_query(self, sentence):
        tokens = word_tokenize(sentence)
        tags = pos_tag(tokens)
        stems = [self.lemmatizer.lemmatize(i) for i in tokens]
        x = self.encode(tags, stems)
        return [x]
