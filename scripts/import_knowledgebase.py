import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s]:%(asctime)s:%(message)s")
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import shutil
import re
import string
from tqdm import tqdm
from exceptions import KnowledgeBaseParsingError
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--model', '-M', help="Choose which model to train the classifier with", default='albert', type=str)
args = parser.parse_args()

def _check_usability():
    if not os.path.exists('../model/'):
        os.mkdir('../model/')
    if not os.path.exists('../model/nlu'):
        os.mkdir('../model/nlu')
    if not os.path.exists('../model/vision'):
        os.mkdir('../model/vision')

_check_usability()

puncts = re.compile('[%s]' % re.escape(string.punctuation))

file = input("Please enter the knowledgebase file you want to import: ")
default = {
"action": None,
"query": [],
"entity": [],
"response": [],
"webhook": False
}
logging.info("Training model set to {}".format(args.model.upper()))

if file.endswith(".txt"):
    logging.info("Selected file with a Text (.txt) Format")
    prefix = file.split("/")[-1].strip(".txt")
    logging.info("Parsing the file")
    with open(file, "r") as f:
        data = f.readlines()
    if len(data)%2:
        raise KnowledgeBaseParsingError("Error in parsing the KnowledgeBase. KnowledgeBase should contain query and response in consecutive lines and no blank lines. There should be exactly one response for exactly one question")
    for i in tqdm(range(len(data))):
        if i%2==0:
            intent = default.copy()
            intent['action'] = prefix+"."+data[i].replace(" ","_")
            intent['query'] = [puncts.sub("", data[i].strip("\n").lower())]
        else:
            intent['response'] = [data[i].strip("\n")]
            try:
                os.mkdir("../intents/"+intent['action'])
            except FileExistsError:
                shutil.rmtree("../intents/"+intent['action'])
                os.mkdir("../intents/"+intent['action'])
            with open("../intents/"+intent['action']+"/intent.tarjani", "w") as f:
                json.dump(intent, f)
    logging.info("Data imported from KnowledgeBase. Starting agent training")
    os.system("python3 train_after_import.py --shuffle True --model "+args.model)

elif file.endswith(".json"):
    logging.info("Selected file with a JSON (.json) Format")
    prefix = file.split("/")[-1].strip(".json")
    logging.info("Parsing the file")
    with open(file, "r") as f:
        data = json.load(f)
    if len(data['query']) != len(data['response']):
        raise KnowledgeBaseParsingError("Error in parsing the KnowledgeBase. KnowledgeBase should contain exactly one response for each queries")
    for i in tqdm(range(len(data['query']))):
        intent = default.copy()
        intent['action'] = prefix+"."+data['query'][i].replace(" ","_")
        intent['query'] = [puncts.sub("", data['query'][i].lower())]
        intent['response'] = [data['response'][i]]
        try:
            os.mkdir("../intents/"+intent['action'])
        except FileExistsError:
            shutil.rmtree("../intents/"+intent['action'])
            os.mkdir("../intents/"+intent['action'])
        with open("../intents/"+intent['action']+"/intent.tarjani", "w") as f:
            json.dump(intent, f)
    logging.info("Data imported from KnowledgeBase. Starting agent training")
    os.system("python3 train_after_import.py --shuffle True --model "+args.model)


elif file.endswith(".csv"):
    logging.info("Selected file with a CSV (.csv) Format")
    try:
        import pandas as pd
    except ImportError:
        logging.warn("Please install Pandas to import CSV files. You can install it by entering 'pip3 install pandas' in the terminal")
        exit()
    separator = input("Enter the separator used in the file. Leave empty to use default ',': ")
    if not separator:
        separator = ','
    logging.info("Parsing the file")
    prefix = file.split("/")[-1].strip(".csv")
    data = pd.read_csv(file, sep=separator)
    if len(data['query']) != len(data['response']):
        raise KnowledgeBaseParsingError("Error in parsing the KnowledgeBase. KnowledgeBase should contain exactly one response for each queries")
    query = data['query'].tolist()
    response = data['response'].tolist()
    for i in tqdm(range(len(data['query']))):
        intent = default.copy()
        intent['action'] = prefix+"."+data['query'][i].replace(" ","_")
        intent['query'] = [puncts.sub("", data['query'][i].lower())]
        intent['response'] = [data['response'][i]]
        try:
            os.mkdir("../intents/"+intent['action'])
        except FileExistsError:
            shutil.rmtree("../intents/"+intent['action'])
            os.mkdir("../intents/"+intent['action'])
        with open("../intents/"+intent['action']+"/intent.tarjani", "w") as f:
            json.dump(intent, f)
    logging.info("Data imported from KnowledgeBase. Starting agent training")
    os.system("python3 train_after_import.py --shuffle True --model "+args.model)


else:
    print("The provided file format is not yet supported. You can submit a feedback or contribute to add your required file formats")
