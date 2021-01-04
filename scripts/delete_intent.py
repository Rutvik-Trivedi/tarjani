import os
import pickle
import shutil
from argparse import ArgumentParser
import logging
logging.basicConfig(level=logging.INFO,format="[%(levelname)s]:%(asctime)s:%(message)s")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from trainer import Trainer


def delete(model='lstm', train=True):
    os.system('clear')
    l = os.listdir('../intents/')
    intent_name = input("\nEnter the name of the intent to delete: ")
    if intent_name not in l:
        os.system('clear')
        print("Intent does not exist. Please enter a valid name")
        delete()
    logging.info("Deleting the intent...")
    shutil.rmtree("../intents/"+intent_name)
    if not train:
        logging.warn('Intent deleted successfully. Train option set to False. To train the agent, please run train_after_import.py file')
        return
    logging.info("Intent deleted. Starting agent training...")
    trainer = Trainer()
    trainer.train_intent(train_model=model)
    logging.info("Agent training completed. Intent removed successfully")

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--model', '-M', type=str, default='lstm', help="Choose which model to train the classifier on after deleting the intent. Default is LSTM")
    parser.add_argument('--train', '-t'. type=bool, default=Truem help="Whether to train the agent after creating the intent or not")
    return parser

if __name__=='__main__':
    parser = get_parser()
    args = parser.parse_args()
    delete(model=args.model, train=args.train)
