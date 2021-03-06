import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s]:%(asctime)s:%(message)s")
from argparse import ArgumentParser
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from trainer import Trainer

parser = ArgumentParser()
parser.add_argument("--mode", "-m", help="Which mode is the script run on", default="import", type=str)
parser.add_argument("--shuffle", "-s", help="Whether to shuffle the intent data", default=False, type=bool)
parser.add_argument('--model', "-M", help="Choose which model to train the classifier on", default='lstm', type=str)
args = parser.parse_args()


trainer = Trainer()
logging.info("Starting agent training after {}ing...".format(args.mode))
trainer.train_intent(shuffle=args.shuffle, train_model=args.model)
logging.info("Agent training complete. Agent successfully {}ted".format(args.mode))
