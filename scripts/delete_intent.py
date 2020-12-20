import os
import pickle
import shutil
from argparse import ArgumentParser
import logging
logging.basicConfig(level=logging.INFO,format="[%(levelname)s]:%(asctime)s:%(message)s")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from trainer import Trainer

def _check_usability():
    if not os.path.exists('../model/'):
        os.mkdir('../model/')
    if not os.path.exists('../model/nlu'):
        os.mkdir('../model/nlu')
    if not os.path.exists('../model/vision'):
        os.mkdir('../model/vision')

def delete(model='lstm'):
    _check_usability()
    os.system('clear')
    l = os.listdir('../intents/')
    intent_name = input("\nEnter the name of the intent to delete: ")
    if intent_name not in l:
        os.system('clear')
        print("Intent does not exist. Please enter a valid name")
        delete()
    logging.info("Deleting the intent...")
    shutil.rmtree("../intents/"+intent_name)
    logging.info("Intent deleted. Starting agent training...")
    trainer = Trainer()
    trainer.train_intent(train_model=model)
    logging.info("Agent training completed. Intent removed successfully")

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--model', '-M', type=str, default='lstm', help="Choose which model to train the classifier on after deleting the intent. Default is LSTM")
    return parser

if __name__=='__main__':
    parser = get_parser()
    args = parser.parse_args()
    delete(model=args.model)
