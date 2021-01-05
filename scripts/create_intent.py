import os
import json
import argparse
from trainer import Trainer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create(edit=False, intent_folder='../intents', model='lstm', train=True):

    os.system('clear')
    l = os.listdir('../intents/')
    print("***Welcome to the TARJANI Intent Creator***")
    print()
    intent_name = input("Enter the name of the intent: ")
    if intent_name in l and not edit:
        os.system('clear')
        print("Intent of the same name already exists. Please choose a unique name")
        create()
    print()
    intent = {
    'action': intent_name,
    'query': [],
    'entity': [],
    'response': [],
    }
    flag = input("Are there any User-Defined Entities for this intent (Y/n)? : ")
    if flag.lower()=='y':
        print("\n***Your intent contains User-Defined Entities. Please provide as many training phrases as possible for better results***\n")
        print("Enter the name of the Entities separated by commas. Please do not include any blank spaces: ", end='')
        entity = input().split(',')
    print("Enter the query phrases for the intent. Enter '-1' to stop")
    question = None
    while question!='-1':
        question = input("Enter query: ")
        if question=='-1':
            break
        if question:
            intent['query'].append(question)
        if flag.lower()=='y':
            endict = {}
            for i in entity:
                index = input("\tEnter the index (or index range separated by ':') where {} entity is present. Index starts from 0: ".format(i))
                if ':' in index:
                    index = index.split(":")
                    endict[i] = list(range(int(index[0]), int(index[1])+1))
                else:
                    try:
                        endict[i] = int(index)
                    except ValueError:
                        endict[i] = None
            intent['entity'].append(endict)
    print()
    print("Enter the responses you would like TARJANI to give you. Enter -1 to stop")
    question = None
    while question!='-1':
        question=input("Response: ")
        if question=='-1':
            break
        intent['response'].append(question)
    print()
    sudo = input("Shall TARJANI call any webhook (Y/n)? : ")
    if sudo.lower()=='n':
        intent['webhook'] = False
    else:
        intent['webhook'] = True
        intent['url'] = input("Enter the full webhook URL: ")

    os.mkdir(intent_folder+'/'+intent_name)

    flag = input("Will you require a skill.py for creating a skill (Y/n)? : ")
    if flag.lower() == 'y':
        with open(intent_folder+'/'+intent_name+'/skill.py', 'w') as f:
            print('Initialized a blank skill.py file for creating the skill of this intent')
            pass
    print("Creating intent...")
    with open(intent_folder+'/'+intent_name+'/intent.tarjani', 'w') as f:
        json.dump(intent, f)
    if not train:
        print("Intent created successfully. Training option set to False. To train the agent, please run train_after_import.py file")
        return
    print("Intent created. Starting Agent Training...")
    trainer = Trainer(pipeline_name=args.model)
    trainer.train_intent()
    if flag.lower()=='y':
        print("Sorting out the Entities...")
        trainer.train_entity(intent_name)
        print("Done")
    print("Agent training completed. Intent added successfully")
    return

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edit', '-e', type=bool, default=False, help="Whether to use in edit mode or not")
    parser.add_argument('--model', '-M', help="Choose which model to train the classifier on. Default is LSTM", default='lstm', type=str)
    parser.add_argument('--train', '-t', type=bool, default=True, help="Whether to train the agent after creating the intent or not")
    return parser

if __name__=='__main__':
    parser = get_parser()
    args = parser.parse_args()
    create(edit=args.edit, model=args.model, train=args.train)
