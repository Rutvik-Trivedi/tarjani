from argparse import ArgumentParser
import json
import os
import pickle
import random
import requests
import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s]:%(asctime)s:%(message)s")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from warnings import warn
from warning import ModelNotTrainedWarning

logging.info("Initializing TARJANI")
url = 'https://tarjani-github.000webhostapp.com/log/'
post_data = {'query': None}        ## Only the query entered will be sent to the server. Please remove these lines and the lines below to stop the logging

from utils import encode
from models import ModelCreator

from nltk import pos_tag
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

import tensorflow as tf
from tensorflow import cast, string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def _check_usability():
    if not os.path.exists('../model/'):
        os.mkdir('../model/')
    if not os.path.exists('../model/nlu'):
        os.mkdir('../model/nlu')
    if not os.path.exists('../model/vision'):
        os.mkdir('../model/vision')

def _warn():
    if sorted(os.listdir('../intents/')) == sorted(['welcome', 'fallback']):
        warn("Seems that the model is not trained properly. Results might not be good. Please create an intent first", ModelNotTrainedWarning)

def predict(train_X, model):
    train_X = cast([train_X], string)
    @tf.function(experimental_relax_shapes=True)
    def _albert_predict(train_X, model):
        return model(train_X)
    outputs = _albert_predict(train_X, model)
    return outputs


_check_usability()

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
if os.path.isdir('../model/nlu/default.tarjani'):
    trained_model = 'primitive'
    model = load_model('../model/nlu/default.tarjani')
else:
    trained_model = 'modern'
    with open('../model/nlu/default.tarjani', 'rb') as f:
        model = pickle.load(f)
    creator = ModelCreator()
    featurizer = creator.make_albert()
with open('../model/nlu/settings.tarjani', 'rb') as f:
    tokenizer, classes = pickle.load(f)
maxlen = classes.pop()
with open('../model/nlu/ner.tarjani', 'rb') as f:
    general = pickle.load(f)


def interact(sensitivity):
    os.system('clear')
    print("***Welcome to the TARJANI Interactive Mode***")
    print("Press Ctrl+C to exit\n")
    _warn()
    while True:
        try:
            response = {}
            question = input("\nEnter your query: ")
            post_data['query'] = question                   # Assigns the question to the post data
            tokens = word_tokenize(question)
            tags = pos_tag(tokens)
            lemma = [lemmatizer.lemmatize(i.lower()) for i in tokens]
            stems = [stemmer.stem(i) for i in tokens]
            if trained_model == 'primitive':
                sentence = ' '.join(lemma)
                sequences = tokenizer.texts_to_sequences([question])
                pad_seq = pad_sequences(sequences, maxlen=maxlen, padding='post')
                output = model.predict(pad_seq)[0]
            else:
                pad_seq = predict(question, featurizer)
                output = model.predict_proba(pad_seq)[0]
            if output.max()>=sensitivity:
                result = classes[output.argmax()]
            else:
                result = 'fallback'
            response['action'] = result
            print("Detected Intent: "+result)
            with open('../intents/'+result+'/intent.tarjani', 'r') as f:
                data = json.load(f)
            try:
                response['webhook'] = data['webhook']
                response['url'] = data['url']
            except KeyError:
                pass
            if data['response']:
                response['response'] = random.choice(data['response'])
                print(response['response'])
            inp = encode(tags, stems)
            entity = {}
            if os.path.exists('../intents'+'/'+result+'/entity.tarjani'):
                with open('../intents'+'/'+result+'/entity.tarjani', 'rb') as f:
                    crf = pickle.load(f)
                output = crf.predict([inp])[0]
                for i in range(len(output)):
                    if output[i]!='O':
                        try:
                            entity[output[i]] = entity[output[i]] + " " + tokens[i]
                        except:
                            entity[output[i]] = tokens[i]
            output2 = general.predict([inp])[0]
            for i in range(len(output2)):
                if output2[i]!='O':
                    entity[output2[i]] = tokens[i]
            response['entity'] = entity
            print(response)
            if os.path.exists('../intents'+'/'+result+'/skill.py'):
                command = 'python3 '+'../intents/'+result+'/skill.py '+"'"+json.dumps(response)+"'"
                os.system(command)

            ## Please remove this line below to stop sending anonymous usage logs
            ## In case the logging process makes the software slow, or you do not want to contribute to improving TARJANI,
            ## please remove this line and it will stop TARJANI from logging the queries. Or you can just disconnect your internet connection :)
            #
            try:
                response = requests.post(url=url, data=post_data)
            except:
                pass
            #
            #
            ## Removing the above line stops TARJANI from logging the info. Thank you for contributing to the developement of TARJANI


        except KeyboardInterrupt:
            print("\n\nThank you for using TARJANI and contributing to its developement. Please open an issue or submit a feedback to help improve TARJANI")
            return 1

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--sensitivity', '-s', help="Sensitivity of the model, defaults to 0.6", default=0.6, type=float)
    return parser

if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()
    interact(args.sensitivity)
