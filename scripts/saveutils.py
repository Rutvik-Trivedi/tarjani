import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
import pickle
import dill
import shutil
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

def _safe_delete(path):
    try:
        shutil.rmtree(path)
    except NotADirectoryError:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    except FileNotFoundError:
        pass


def save_intent_pipeline(name, pipeline, settings, folder='../model/nlu/'):

    pipeline = _prepare_intent_pipeline_for_saving(pipeline)

    for type, model in pipeline.items():
        if model.requires_save:
            _safe_delete(folder+type+'/'+name+'_'+type+'.tarjani')
            model.save(folder+type+'/'+name+'_'+type+'.tarjani')

    with open(folder+'settings/'+name+'_settings.tarjani', 'wb') as f:
        pickle.dump(settings, f)


def save_entity_pipeline(name, intent_name, pipeline, settings, folder='../intents/'):

    pipeline = _prepare_entity_pipeline_for_saving(pipeline)

    for type, model in pipeline.items():
        if model.requires_save:
            _safe_delete(folder+intent_name+'/'+name+'_'+type+'.tarjani')
            model.save(folder+intent_name+'/'+name+'_'+type+'.tarjani')

    with open(folder+intent_name+'/'+name+'_'+'_settings.tarjani', 'wb') as f:
        pickle.dump(settings, f)
