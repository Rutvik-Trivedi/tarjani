import shutil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s]:%(asctime)s:%(message)s")

flag = input("\nAre you sure to reset your model to Blank? (Blank model contains the two default intents: welcome and fallback) This action is not reversible (Y/n): ")

if flag.lower() == 'n':
    logging.info("Abort")

else:
    l = os.listdir('../intents')
    for i in l:
        if i not in ['fallback', 'welcome']:
            shutil.rmtree("../intents/"+i)

    os.system('python3 train_after_import.py --mode reset')
