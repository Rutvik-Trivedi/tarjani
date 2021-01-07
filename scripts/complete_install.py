import os
import logging
import tarfile
logging.basicConfig(level=logging.INFO, format="[%(levelname)s]:%(asctime)s:%(message)s")
import warnings


logging.info('Completing the installation process')

logging.info('Installing Github Clone for downloading addon models')
os.system('pip3 install git+git://github.com/HR/github-clone#egg=ghclone > logs.txt')
logging.info("Done installing Github Clone.")
logging.info("Trying to install the required dependencies using Pip. This may take a while based on the speed of your network connection")
os.system('pip3 install -r ../requirements.txt > logs.txt')
os.remove('logs.txt')

if not os.path.exists('../intents'):
    os.mkdir('../intents')
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

if not os.path.exists('../model/nlu/ner.tarjani'):
    logging.error('The General Mandatory NER Model needs to be downloaded. Please paste this link in your browser to download the file\nhttp://tarjani.is-great.net/download/index.php?q=ner.tarjani')
    exit()

if not os.path.exists('../model/nlu/glove'):
    logging.error('GloVe Embedding file unavailable. Navigate to the tarjani/model/nlu folder and run the command:\nghclone https://github.com/Rutvik-Trivedi/tarjani-model-zoo/tree/main/glove\nand then run this script again')
    exit()

if not os.path.exists('experimental/data'):
    os.mkdir('experimental/data')

logging.info("All mandatory files and directory structures found in place")
if not os.path.exists('../model/nlu/glove/glove.6B.50d.txt'):
    logging.info("Extracting the embedding file")
    tarball = tarfile.open('../model/nlu/glove/glove.6B.50d.tar.xz')
    tarball.extractall(path='../model/nlu/glove/')
    tarball.close()

logging.info('Starting initial agent training')
os.system('python3 train_after_import.py')
os.system('clear')
logging.info('Installation Complete. Thank you for trying out TARJANI. You may start by creating an intent first')
