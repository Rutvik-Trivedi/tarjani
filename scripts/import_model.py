import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s]:%(asctime)s:%(message)s")
import argparse
import tarfile
import os
import shutil
from glob import glob
import warnings
warnings.filterwarnings('ignore')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, required=True, help="Name/Path of the model")
    parser.add_argument('--model', '-m', type=str, default='lstm', help="Model to train the agent after importing")
    return parser

def remove_trained_models(intent_name):
    l = os.listdir('intents/'+intent_name)
    for file in l:
        if file.endswith("entity.tarjani"):
            os.remove('intents/'+intent_name+'/'+file)

def main():
    parser = get_parser()
    args = parser.parse_args()

    path = args.name
    model = args.model
    name = path.rstrip('.tarjani')

    logging.info("Extracting the model file information")
    tarball = tarfile.open(name+'.tarjani')
    tarball.extractall()
    tarball.close()

    logging.info("Sorting the model files and cleaning up")
    intents = os.listdir('intents/')
    for intent in intents:
        remove_trained_models(intent)
        try:
            shutil.rmtree('../intents/'+intent)
        except FileNotFoundError:
            pass
        shutil.move('intents/'+intent, '../intents', copy_function=shutil.copytree)
    shutil.rmtree('intents/')

    os.system('python3 train_after_import.py --model {}'.format(model))

if __name__ == '__main__':
    main()
