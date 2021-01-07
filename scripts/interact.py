import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s]:%(asctime)s:%(message)s")
logging.info("Initializing TARJANI")

import argparse
import os
import loadutils
import requests
import random
import tarfile
from time import sleep

import warnings
warnings.filterwarnings('ignore')

import json
from nltk.tokenize import word_tokenize


def _warn():
    if sorted(os.listdir('../intents/')) == ['fallback', 'welcome']:
        logging.warning("  ModelNotTrainedWarning: Seems that the model is not trained properly. Results might not be good. Please create an intent first.")
        flag = input('Force initialize TARJANI anyway? (Y/n): ').lower()
        if flag == 'n':
            exit()
        logging.info('Forcing initialization')


def _check_usability():
    if not os.path.exists('../model'):
        os.mkdir('../model')
    if not os.path.exists('../model/nlu'):
        os.mkdir('../model/nlu')
    if not os.path.exists('../model/vision'):
        os.mkdir('../model/vision')
    if not os.path.exists('../model/nlu/tokenizer'):
        os.mkdir('../model/nlu/tokenizer')
    if not os.path.exists('../model/nlu/featurizer'):
        os.mkdir('../model/nlu/featurizer')
    if not os.path.exists('../model/nlu/classifier'):
        os.mkdir('../model/nlu/classifier')
    if not os.path.exists('../model/nlu/settings'):
        os.mkdir('../model/nlu/settings')
    if not os.path.exists('../model/nlu/glove/glove.6B.50d.txt'):
        _extract_embeddings()
    assert os.path.exists('../model/nlu/glove/glove.6B.50d.txt'), "Error in extracting the embedding file"
    assert os.path.exists('../model/nlu/ner.tarjani'), "NER model file does not exist. Please visit http://tarjani.is-great.net for installation steps"


def _extract_embeddings():
    logging.info("Extracting the embedding file")
    tarball = tarfile.open('../model/nlu/glove/glove.6B.50d.tar.xz')
    tarball.extractall(path='../model/nlu/glove/')
    tarball.close()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensitivity', '-s', type=float, help='Sets the sensitivity of the algorithm. Less sensitivity leads to more lenient model. Default is 0.6', default=0.6)
    parser.add_argument('--model', '-m', type=str, help='Sets the pipeline of the agent. Make sure you have trained this pipeline first. Default is LSTM', default='lstm')
    return parser

def detect_intent(output, sensitivity, classes):
    if output.max()>=sensitivity:
        result = classes[output.argmax()]
    else:
        result = 'fallback'
    return result

def populate_entity(entity, output, sentence):
    sentence = word_tokenize(sentence)
    for i in range(len(output)):
        if output[i]!='O':
            try:
                entity[output[i]] = entity[output[i]] + " " + sentence[i]
            except:
                entity[output[i]] = sentence[i]
    return entity


def interact(sensitivity, pipeline_name):
    escape = ['-1', 'stop', '/exit', 'exit', 'quit']
    url = 'https://tarjani-github.000webhostapp.com/log/'    ## Only the query entered will be sent to the server.
    post_data = {'query': None}                              ## Please remove these lines or the lines below to stop the logging

    logging.info('Initializing the pipeline. Selected pipeline name: {}'.format(pipeline_name.upper()))
    intent_pipeline, intent_settings = loadutils.load_intent_pipeline(pipeline_name)
    classes = intent_settings['encoder'].classes_
    ner = loadutils.load_ner_pipeline()

    os.system('clear')
    print("***Welcome to the TARJANI Interactive Mode***")
    print("Press Ctrl+C to exit\n")

    while True:
        try:
            response = {}
            question = input('\nEnter your query: ').lower()
            if question in escape:
                break
            try:
                post_data['query'] = question
            except:
                pass
            tmp = intent_pipeline['dataloader'].prepare_query(question)
            tmp = intent_pipeline['tokenizer'].prepare_query(tmp, **intent_settings)
            tmp = intent_pipeline['featurizer'].predict(tmp)
            output = intent_pipeline['classifier'].predict(tmp)[0]
            result = detect_intent(output, sensitivity, classes)
            response['action'] = result
            print("Detected Intent: "+result)
            with open('../intents/'+result+'/intent.tarjani', 'r') as f:
                data = json.load(f)
            response['webhook'] = data.get('webhook', False)
            response['url'] = data.get('url', None)
            if data['response']:
                response['response'] = random.choice(data['response'])
                print("TARJANI:", response['response'])

            entity = {}
            tmp = ner['entity_loader'].prepare_query(question)
            output = ner['entity_extractor'].predict(tmp)[0]
            entity = populate_entity(entity, output, question)


            entity_pipeline, entity_settings = loadutils.load_entity_pipeline(pipeline_name, result)
            if entity_pipeline != None:
                entity_pipeline, entity_settings = loadutils.load_entity_pipeline(pipeline_name, result)
                output = entity_pipeline['entity_extractor'].predict(tmp)[0]
                entity = populate_entity(entity, output, question)
                response['entity'] = entity
            else:
                response['entity'] = {}

            print(response)

            if os.path.exists('../intents'+'/'+result+'/skill.py'):
                command = 'python3 '+'../intents/'+result+'/skill.py '+"'"+json.dumps(response)+"'"
                os.system(command)


            ## Please remove these lines below to stop sending anonymous usage logs
            ## In case the logging process makes the software slow, or you do not want to contribute to improving TARJANI,
            ## please remove these lines and it will stop TARJANI from logging the queries. Or you can just disconnect your internet connection :)
            #
            try:
                requests.post(url=url, data=post_data)
            except:
                pass
            #
            #
            ## Removing the above lines stops TARJANI from logging the info. Thank you for contributing to the developement of TARJANI


        except KeyboardInterrupt:
            break

    print("\n\nThank you for using TARJANI and contributing to its developement. Please open an issue or submit a feedback to help improve TARJANI")


if __name__ == '__main__':
    parser = get_parser()
    _check_usability()
    _warn()
    args = parser.parse_args()
    interact(args.sensitivity, args.model)
