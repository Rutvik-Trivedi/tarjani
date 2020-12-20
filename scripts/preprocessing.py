import random

from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag
from nltk.tokenize import word_tokenize

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow import cast, string
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np

from sklearn.preprocessing import LabelEncoder

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Preprocessor():

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.punctuations = "\"'?!@#$%^*()+_-"

    @tf.function(experimental_relax_shapes=True)
    def _albert_predict(self, train_X, model):
        return model(train_X)

    def make_permutations(self, sentence, action, shuffle):
        for x in self.punctuations:
            sentence = sentence.replace(x,'')
        tokens = word_tokenize(sentence)
        lemmatized_tokens = [self.lemmatizer.lemmatize(i.lower()) for i in tokens]
        return_list= []
        if not shuffle:
            return_list.append(' '.join(lemmatized_tokens))
            action_list = [action]
        else:
            for i in range(5):
                random.shuffle(lemmatized_tokens)
                return_list.append(" ".join(lemmatized_tokens))
            action_list = [action]*5
        return (return_list, action_list)

    def create_tokenizer(self, train_X):
        maxlen = -1
        for i in train_X:
            if len(i.split()) > maxlen:
                maxlen = len(i.split())
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train_X)
        sequences = tokenizer.texts_to_sequences(train_X)
        pad_seq = pad_sequences(sequences, maxlen=maxlen, padding='post')
        return (tokenizer, pad_seq, maxlen)

    def create_embedding_matrix(self, vocab_size, tokenizer, num_layers=50):
        embedding_vector = {}
        f = open('../glove/glove.6B.50d.txt')
        for line in f:
            value = line.split(' ')
            word = value[0]
            coef = np.array(value[1:],dtype = 'float32')
            embedding_vector[word] = coef

        embedding_matrix = np.zeros((vocab_size,num_layers))
        for word,i in tokenizer.word_index.items():
                embedding_value = embedding_vector.get(word)
                if embedding_value is not None:
                    embedding_matrix[i] = embedding_value
        f.close()
        return embedding_matrix

    def create_albert_data(self, train_X, model):
        train_X = cast(train_X, string)
        outputs = self._albert_predict(train_X, model)
        return outputs
