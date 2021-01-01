import os
import logging
import tarfile
logging.basicConfig(level=logging.INFO, format="[%(levelname)s]:%(asctime)s:%(message)s")
import warnings
from warning import *


logging.info('Completing the installation process')

if not os.path.exists('../model'):
    os.mkdir('../model')
if not os.path.exists('../model/nlu'):
    os.mkdir('../model/nlu')
if not os.path.exists('../model/vision'):
    os.mkdir('../model/vision')
if not os.path.exists('../model/nlu/ner.tarjani'):
    logging.error('The General Mandatory NER Model needs to be downloaded. Please paste this link in your browser to download the file\nhttp://tarjani.is-great.net/download/index.php?q=ner.tarjani')
if not os.path.exists('experimental/data'):
    os.mkdir('experimental/data')

logging.info("All mandatory files and directory structures found in place")
if not os.path.exists('../glove/glove.6B.50d.txt'):
    logging.info("Extracting the embedding file")
    tarball = tarfile.open('../glove/glove.6B.50d.tar.xz')
    tarball.extractall(path='../glove/')
    tarball.close()

logging.info('Installing Github Clone for downloading addon models')
os.system('pip3 install git+git://github.com/HR/github-clone#egg=ghclone > logs.txt')
logging.info("Done installing Github Clone.")
logging.info("Trying to install the required dependencies using Pip. This may take a while based on the speed of your network connection")
os.system('pip3 install -r ../requirements.txt > logs.txt')
os.remove('logs.txt')
logging.info("Done installing the required dependencies. Thank you for your patience")
logging.info('Installation Complete. Thank you for trying out TARJANI. You may start by creating an intent first')
