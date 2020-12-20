import tensorflow as tf
import tensorflow_text
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Input, Flatten, Conv1D, MaxPooling1D
import tensorflow_hub as hub
import sklearn_crfsuite
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ModelCreator():

    def __init__(self):
        pass

    def get_kernel_size(self, maxlen):
        if maxlen==1 or maxlen==2:
            return 1
        else:
            return int(maxlen/3)

    def make_lstm(self, maxlen, vocab_size, embedding_matrix, num_classes, num_layers=50, dropout=0.2, activation='softmax', trainable_embeddings=False):
        model = Sequential()
        model.add(Input(shape=(maxlen,)))
        model.add(Embedding(vocab_size,num_layers,weights = [embedding_matrix],input_length=maxlen,trainable = trainable_embeddings))
        model.add(LSTM(num_layers))
        model.add(Dropout(dropout))
        model.add(Flatten())
        model.add(Dense(num_classes, activation=activation))
        return model

    def make_cnn(self, maxlen, vocab_size, embedding_matrix, num_classes, num_layers=50, filters=32, activation='relu', classifier_activation='softmax', pool_size=2, trainable_embeddings=False):
        model = Sequential()
        model.add(Input(shape=(maxlen,)))
        model.add(Embedding(vocab_size,num_layers,weights = [embedding_matrix],input_length=maxlen,trainable = trainable_embeddings))
        model.add(Conv1D(filters=filters, kernel_size=self.get_kernel_size(maxlen), activation=activation))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(Flatten())
        model.add(Dense(10, activation=activation))
        model.add(Dense(num_classes, activation=classifier_activation))
        return model

    def make_crf(self, algorithm, c1, c2, max_iterations, all_possible_transitions):
        crf = sklearn_crfsuite.CRF(
            algorithm=algorithm,
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=all_possible_transitions
            )
        return crf

    def make_albert(self, trainable_albert=False, model_path='../model/nlu/albert/albert_model', tokenizer_path='../model/nlu/albert/albert_tokenizer'):
        input_ = tf.keras.layers.Input(shape=(), dtype=tf.string)
        albert = hub.KerasLayer(model_path)
        tokenizer = hub.KerasLayer(tokenizer_path)
        processed = tokenizer(input_)
        output = albert(processed)['pooled_output']
        model = tf.keras.models.Model(inputs=input_, outputs=output)
        return model

    def make_svc(self, defaults = None):
        if defaults is None:
            defaults = {
            "C": [1, 2, 5, 10, 20, 100],
            "gamma": [0.1],
            "kernel": ["linear"],
            }
        svc= SVC(probability=True)
        clf = GridSearchCV(svc, defaults)
        return clf
