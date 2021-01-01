import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import string
import re
import json
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.preprocessing import LabelEncoder


class BaseLoader:
    '''
    Similar to the Torch Dataloader class.
    You will need to inherit this class while creating a new
    dataloader architecture.
    '''

    def __init__(self, **kwargs):
        self.stopwords = stopwords.words('english')
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.puncts = re.compile('[%s]' % re.escape(string.punctuation))
        self.encoder = LabelEncoder()
        self.batch_size = kwargs.get('batch_size', 32)

    def data(self):
        raise NotImplementedError

    def name(self):
        return 'base_loader'

    def _intent_folder(self):
        return '../intents/'

    def _get_intent_names(self):
        intents = os.listdir(self._intent_folder())
        return list(set(intents) - set(['fallback']))

    def _permute_batch(self, x, y, permute_count=5):
        perms = []
        ret_y = []
        for tokens, target in zip(x,y):
            for _ in range(permute_count):
                tmp = tokens.copy()
                random.shuffle(tmp)
                perms.append(tmp)
                ret_y.append(target)
        return perms, ret_y

    def _tokenize_batch(self, x):
        for i in range(len(x)):
            x[i] = word_tokenize(x[i])
        return x

    def _remove_stopwords_batch(self, x):
        for i in range(len(x)):
            tmp = []
            for j in x[i]:
                if j not in self.stopwords:
                    tmp.append(j)
            x[i] = tmp
        return x

    def _lemmatize_batch(self, x):
        for i in range(len(x)):
            x[i] = [self.lemmatizer.lemmatize(j) for j in x[i]]
        return x

    def _stem_batch(self, x):
        for i in range(len(x)):
            x[i] = [self.stemmer.stem(j) for j in x[i]]
        return x

    def _pos_tag_batch(self, x):
        for i in range(len(x)):
            x[i] = pos_tag(x[i])
        return x[i]

    def _join_batch(self, x):
        return [' '.join(i) for i in x]

    def _remove_puncts_batch(self, x):
        for i in range(len(x)):
            x[i] = self.puncts.sub("", x[i])
        return x

    def _get_raw_data(self):
        folder = self._intent_folder()
        intents = self._get_intent_names()
        ret = []
        y = []

        for intent in intents:
            if intent == 'fallback':
                continue
            with open(folder+intent+'/intent.tarjani', 'r') as f:
                data = json.load(f)
            ret += data['query']
            y += [data['action']] * len(data['query'])

        return ret, y

    def _class_encode(self, y, fit=True):
        y = list(y)
        if fit:
            return self.encoder.fit_transform(y)
        else:
            return self.encoder.transform(y)
