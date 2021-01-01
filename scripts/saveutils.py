import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
import dill
import tensorflow as tf


def _prepare_intent_pipeline_for_saving(pipeline):
    return {
        'tokenizer': pipeline[0],
        'featurizer': pipeline[1],
        'classifier': pipeline[2]
    }

def _prepare_entity_pipeline_for_saving(pipeline):
    return {
        'entity': pipeline[0]
    }


def save_intent_pipeline(name, pipeline, settings, folder='../model/nlu/'):
    pipeline = _prepare_intent_pipeline_for_saving(pipeline)

    for type, model in pipeline.items():
        model.save(folder+name+'_'+type+'.tarjani')

    with open(folder+name+'_settings.tarjani', 'wb') as f:
        pickle.dump(settings, f)


def save_entity_pipeline(name, intent_name, pipeline, settings, folder='../intents/'):
    pipeline = _prepare_entity_pipeline_for_saving(pipeline)

    for type, model in pipeline.items():
        model.save(folder+intent_name+'/'+name+'_'+type+'.tarjani')

    with open(folder+intent_name+'/'+name+'_'+type+'.tarjani', 'wb') as f:
        pickle.dump(settings, f)
