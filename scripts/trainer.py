import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s]:%(asctime)s:%(message)s")
logging.info('Importing the required files. It may take a while')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import pickle
import json
from tqdm import tqdm

from exceptions import EntityOverwriteError, ModelNotFoundError
from models import ModelCreator
from preprocessing import Preprocessor
from utils import *

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

def _check_usability():
    if not os.path.exists('../model/'):
        os.mkdir('../model/')
    if not os.path.exists('../model/nlu'):
        os.mkdir('../model/nlu')
    if not os.path.exists('../model/vision'):
        os.mkdir('../model/vision')


class Trainer():

    def __init__(self):
        logging.info('Initializing the Trainer')
        _check_usability()
        assert os.path.exists('../glove/glove.6B.50d.txt'), "Embedding file does not exist. Please untar the glove.6B.50d.tar.xz file present in the glove folder to use TARJANI properly"
        assert os.path.exists('../model/nlu/ner.tarjani'), "NER model file does not exist. Please visit http://tarjani.is-great.net for installation steps"
        self.model_creator = ModelCreator()
        self.processor = Preprocessor()
        self.encoder = LabelEncoder()


    def collect_intent_data(self, shuffle, intent_folder='../intents', save=False, save_folder='../data/intent'):
        intent_list = os.listdir(intent_folder)
        train_X = []
        train_y = []

        for intent in tqdm(intent_list):
            if intent=='fallback':
                continue

            filepath = intent_folder+'/'+intent+'/intent.tarjani'
            with open(filepath, 'r') as f:
                data = json.load(f)

            for query in data['query']:
                x, y = self.processor.make_permutations(query, intent, shuffle=shuffle)
                train_X+=x
                train_y+=y

        train_y = self.encoder.fit_transform(train_y)
        if save:
            logging.info('Save option set to true. Saving the data')
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            with open(save_folder+'/data.tarjani', 'wb') as f:
                pickle.dump((train_X, train_y), f)


        return train_X, train_y, len(intent_list)-1, self.encoder.classes_


    def collect_albert_data(self, intent_folder='../intents', save=False, save_folder='../data/intent'):
        intent_list = os.listdir(intent_folder)
        logging.info('Creating ALBERT featurizer. This may take a while')
        albert_model = self.model_creator.make_albert()
        logging.info('Generating features from sentences. Time taken by this depends on the number of intents and total examples')
        train_X = []
        train_y = []

        for intent in tqdm(intent_list):
            if intent=='fallback':
                continue

            filepath = intent_folder+'/'+intent+'/intent.tarjani'
            with open(filepath, 'r') as f:
                data = json.load(f)

            l = len(data['query'])
            train_y+=[intent]*l
            train_X.append(self.processor.create_albert_data(data['query'], albert_model))

        train_X = tf.keras.backend.concatenate(train_X, axis=0)
        train_y = self.encoder.fit_transform(train_y)
        if save:
            logging.info('Save option set to true. Saving the data')
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            with open(save_folder+'/data.tarjani', 'wb') as f:
                pickle.dump((train_X, train_y), f)


        return train_X.numpy(), train_y, self.encoder.classes_



    def train_intent(self, save=True, save_folder='../model/nlu', model_name='default', shuffle=False, train_model='lstm'):
        logging.info('Collecting training data for intents')
        if train_model in ['lstm', 'cnn']:
            train_X, train_y, num_classes, classes = self.collect_intent_data(shuffle=shuffle)
            logging.info('Creating Basic Tokenizer')
            tokenizer, pad_seq, maxlen = self.processor.create_tokenizer(train_X)
            vocab_size = len(tokenizer.word_index)+1
            embedding_matrix = self.processor.create_embedding_matrix(vocab_size, tokenizer)

        else:
            train_X, train_y, classes = self.collect_albert_data()

        logging.info('Training model is set to {}'.format(train_model.upper()))
        if train_model == 'lstm':
            model = self.model_creator.make_lstm(maxlen, vocab_size, embedding_matrix, num_classes)
            model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

        elif train_model == 'cnn':
            model = self.model_creator.make_cnn(maxlen, vocab_size, embedding_matrix, num_classes)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        elif train_model == 'albert':
            model = self.model_creator.make_svc()

        else:
            raise ModelNotFoundError('Could not load the model {}. It is not yet supported. Please open an issue to get your model added.'.format(train_model))

        #Train the model
        logging.info("Started Training. This may take a while if the number of intents is large")
        if train_model in ['lstm', 'cnn']:
            history = model.fit(pad_seq, train_y, epochs=20, batch_size=5, verbose=1)
        elif train_model == 'albert':
            history = model.fit(train_X, train_y)
        if save:
            logging.info('Save model option is set to True. Saving the model')
            classes = classes.tolist()
            try:
                classes.append(maxlen)
            except UnboundLocalError:
                classes.append(None)
            if train_model in ['lstm', 'cnn']:
                model.save(save_folder+'/'+model_name+'.tarjani')
                with open(save_folder+'/settings.tarjani', 'wb') as f:
                    pickle.dump((tokenizer, classes), f)
            else:
                with open(save_folder+'/'+model_name+'.tarjani', 'wb') as f:
                    pickle.dump(model, f)
                with open(save_folder+'/settings.tarjani', 'wb') as f:
                    pickle.dump(([], classes), f)
        return history



    def collect_entity_data(self, intent_name, intent_folder='../intents', save=False):
        train_X = []
        train_y = []
        stemmer = PorterStemmer()
        filepath = intent_folder+'/'+intent_name+'/intent.tarjani'
        with open(filepath, 'r') as f:
            data = json.load(f)
        for i in tqdm(range(len(data['query']))):
            tokens = word_tokenize(data['query'][i])
            tags = pos_tag(tokens)
            stems = [stemmer.stem(i) for i in tokens]
            x = encode(tags, stems)
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

        if save:
            logging.info("Save option set to true. Saving the data for future use")
            with open(intent_folder+'/'+intent_name+'/data.tarjani', 'wb') as f:
                pickle.dump((train_X, train_y), f)

        return train_X, train_y

    def train_entity(self, intent_name, intent_folder='../intents', algorithm='lbfgs', c1=0.6, c2=0.01, max_iterations=100, all_possible_transitions=True):
        logging.info('Preparing the Entity training data')
        train_X, train_y = self.collect_entity_data(intent_name=intent_name)
        logging.info('Creating the CRF Model')
        crf = self.model_creator.make_crf(algorithm=algorithm, c1=c1, c2=c2, max_iterations=max_iterations, all_possible_transitions=all_possible_transitions)
        logging.info("Starting the training")
        crf.fit(train_X, train_y)
        logging.info("Done. Saving the Entity model settings")
        with open(intent_folder+'/'+intent_name+'/entity.tarjani', 'wb') as f:
            pickle.dump(crf, f)
        return True
