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
        if file.endswith('_entity_extractor.tarjani') or file.endswith("_settings.tarjani") or file.endswith("entity.tarjani"):
            os.remove('intents/'+intent_name+'/'+file)

def make_tarfile(output_filename, source_dir='intents'):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

def main():
    parser = get_parser()
    args = parser.parse_args()

    name = args.name.rstrip('.tarjani')
    logging.info("Gathering model information files")
    try:
        shutil.rmtree('intents')
    except FileNotFoundError:
        pass
    shutil.copytree('../intents', 'intents/')

    intents = os.listdir('intents')
    for intent in intents:
        remove_trained_models(intent)

    logging.info('Generating exported model file')
    make_tarfile(name+'.tarjani')

    logging.info('Cleaning up')
    try:
        shutil.rmtree('intents')
    except FileNotFoundError:
        pass
    logging.info("Model '{}' successfully exported".format(name))

if __name__ == '__main__':
    main()
