import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s]:%(asctime)s:%(message)s")
logging.info('Importing the required files. It may take a while')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import pickle

import yaml

from exceptions import EntityOverwriteError, ModelNotFoundError, PipelineNotFoundError
from dataloaders import SimpleLoader, EmbeddingLoader, CRFLoader
from tokenizers import KerasTokenizer
from featurizers import AlbertFeaturizer, EmbeddingFeaturizer
from classifiers import (
    CNNClassifier, SVCClassifier, LSTMClassifier
)
from entity_extractors import CRFClassifier
import saveutils

from lookup import lookup


def _check_usability():
    if not os.path.exists('../model/'):
        os.mkdir('../model/')
    if not os.path.exists('../model/nlu'):
        os.mkdir('../model/nlu')
    if not os.path.exists('../model/vision'):
        os.mkdir('../model/vision')
    assert os.path.exists('../glove/glove.6B.50d.txt'), "Embedding file does not exist. Please untar the glove.6B.50d.tar.xz file present in the glove folder to use TARJANI properly"
    assert os.path.exists('../model/nlu/ner.tarjani'), "NER model file does not exist. Please visit http://tarjani.is-great.net for installation steps"



class Trainer():

    def __init__(self, pipeline_name='lstm'):
        logging.info('Initializing the Trainer')
        _check_usability()
        self.pipeline_name = pipeline_name
        self.lookup = lookup
        logging.info("Generating the specified Pipeline: {}".format(self.pipeline_name))
        self.pipeline = self._get_pipeline(self.pipeline_name)
        logging.info("Loading the pipeline settings")
        self.settings = self._get_settings(self.pipeline_name)['settings']


    def _get_pipeline(self, name):
        with open('pipelines.yml', 'r') as f:
            pipelines = yaml.load(f, Loader=yaml.FullLoader)
        for pipeline in pipelines:
            if pipeline['name'] == name:
                return pipeline
        else:
            raise PipelineNotFoundError('The specified pipeline could not be found. Did you forget to add it in pipelines.yml?')


    def _get_settings(self, name):
        with open('settings.yml', 'r') as f:
            settings = yaml.load(f, Loader=yaml.FullLoader)

        for setting in settings:
            if setting['name'] == name:
                return setting
        else:
            raise PipelineNotFoundError("The specified pipeline could not be found. Did you forget to add it in settings.yml?")


    def train_intent(self, save=True):
        logging.info('Processing the pipeline')
        if self.pipeline['dataloader']:
            logging.info('Generating the intent training data')
            self.dataloader = self.lookup[self.pipeline['dataloader']](**self.settings['intent'])
            train_X, train_y = self.dataloader.data(**self.settings['intent'])
            self.settings['intent'].update(self.dataloader.__dict__)
            num_classes = len(set(train_y))
            self.settings['intent'].update({'num_classes':num_classes})
        else:
            logging.info("Dataloader Component set to None. Skipping...")
            self.dataloader = None
        if self.pipeline['tokenizer']:
            logging.info('Creating the Tokenizer and processing data')
            self.tokenizer = self.lookup[self.pipeline['tokenizer']](**self.settings['intent'])
            self.tokenizer.set_tokenizer()
            train_X = self.tokenizer.tokenize_and_pad(train_X)
            self.settings['intent'].update(self.tokenizer.__dict__)
        else:
            logging.info("Tokenizer Component set to None. Skipping...")
            self.tokenizer = None
        if self.pipeline['featurizer']:
            logging.info('Generating the featurizer model and processing the data')
            self.settings['intent'].update({'tokenizer':self.tokenizer.tokenizer})
            self.featurizer = self.lookup[self.pipeline['featurizer']](**self.settings['intent'])
            self.settings['intent'].update(self.featurizer.__dict__)
            self.featurizer.build(**self.settings['intent'])
            train_X = self.featurizer.predict(train_X)
        else:
            logging.info("Featurizer Component set to None. Skipping...")
            self.featurizer = None
        if self.pipeline['classifier']:
            logging.info('Generating the intent classifier model')
            self.classifier = self.lookup[self.pipeline['classifier']](**self.settings['intent'])
            logging.info('Training the classifier model')
            history = self.classifier.train(train_X, train_y, **self.settings['intent'])
        else:
            logging.info("Classifier Component set to None. Skipping...")
            self.classifier = None

        if save:
            logging.info("Save option set to True. Saving the required components of the Pipeline")
            pipeline = [self.tokenizer, self.featurizer, self.classifier]
            saveutils.save_intent_pipeline(self.pipeline_name, pipeline, self.settings['intent'])

        return history

    def train_entity(self, intent_name, save=True):
        self.settings['entity'].update({'intent':intent_name})
        if self.pipeline['entity_loader']:
            logging.info('Processing entity data')
            self.entity_loader = self.lookup[self.pipeline['entity_loader']](**[self.settings['entity']])
            train_X, train_y = self.entity_loader.data(self.settings['entity'])
            self.settings['entity'].update(self.entity_loader.__dict__)
        else:
            logging.info("Entity Dataloader Component set to None. Skipping...")
            self.entity_loader, train_X, train_y = None, None, None
        if self.pipeline['entity_extractor']:
            logging.info('Generating the entity extraction model')
            self.entity_extractor = lookup[pipeline['entity_extractor']](**self.settings['entity'])
            logging.info('Training the entity extractor model')
            history = self.entity_extractor.train(train_X, train_y)
            self.settings['entity'].update(self.entity_extractor.__dict__)
        else:
            logging.info("Entity Classifier Component set to None. Skipping...")
            self.entity_extractor = None
            return

        if save:
            logging.info("Save option set to True. Saving the required components of the Pipeline")
            pipeline = [self.entity_extractor]
            saveutils.save_entity_pipeline(self.pipeline_name, intent_name, pipeline, self.settings['entity'])

        return history
