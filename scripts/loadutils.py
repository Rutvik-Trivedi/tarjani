import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
import pickle
import dill
import shutil
import tensorflow as tf
import yaml

from exceptions import *

import classifiers
import dataloaders
import entity_extractors
import featurizers
import tokenizers

from lookup import lookup

def _is_component_present(name, pipeline_name):
    return os.path.exists('../model/nlu/'+name+'/'+pipeline_name+'_'+name+'.tarjani')

def _get_pipeline(name):
    with open('pipelines.yml', 'r') as f:
        pipelines = yaml.load(f, Loader=yaml.FullLoader)
    for pipeline in pipelines:
        if pipeline['name'] == name:
            return pipeline
    else:
        raise PipelineNotFoundError('The specified pipeline could not be found. Did you forget to add it in pipelines.yml?')

def _load_intent_settings(pipeline_name):
    try:
        with open('../model/nlu/settings/'+pipeline_name+'_settings.tarjani', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise PipelineNotFoundError('Settings corresponding to the specified Pipeline not found')

def _load_entity_settings(pipeline_name, intent_name):
    try:
        with open('../intents/'+intent_name+'/'+pipeline_name+'_settings.tarjani', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def _load_component(name, pipeline_name, pipeline, settings, intent_name=None):
    if settings is None:
        return None
    component = lookup[pipeline[name]](**settings)
    if intent_name:
        path = '../intents/'+intent_name+'/'+pipeline_name+'_'+name+'.tarjani'
    else:
        path = '../model/nlu/'+name+'/'+pipeline_name+'_'+name+'.tarjani'
    if component.requires_save:
        try:
            component(**settings)
            component.load(path)
        except AttributeError:
            pass
        except FileNotFoundError:
            if intent_name is None:
                raise PipelineNotFoundError('Component {} corresponding the specified Pipeline {} was not found. Please provide a valid pipeline name or train the component first'.format(name, pipeline_name))
            else:
                return None
    else:
        if intent_name is None:
            component(**settings)
        else:
            component(intent_name=intent_name, **settings)
    return component

def load_intent_pipeline(pipeline_name):
    components = ['dataloader', 'tokenizer', 'featurizer', 'classifier']
    pipeline = _get_pipeline(pipeline_name)
    pipeline_dict = {}

    settings = _load_intent_settings(pipeline_name)

    for component in components:
        pipeline_dict[component] = _load_component(component, pipeline_name, pipeline, settings, intent_name=None)

    return pipeline_dict, settings

def load_entity_pipeline(pipeline_name, intent_name):
    components = ['entity_loader', 'entity_extractor']
    pipeline = _get_pipeline(pipeline_name)
    pipeline_dict = {}

    settings = _load_entity_settings(pipeline_name, intent_name)

    for component in components:
        pipeline_dict[component] = _load_component(component, pipeline_name, pipeline, settings, intent_name=intent_name)

    if not any(list(pipeline_dict.values())):
        return None, None

    return pipeline_dict, settings

def load_ner_pipeline():
    components = ['entity_loader', 'entity_extractor']
    pipeline_dict = {
        'entity_loader': lookup['crf_loader'](),
        'entity_extractor': lookup['crf_classifier']()
    }
    pipeline_dict['entity_extractor'].load('../model/nlu/ner.tarjani')
    return pipeline_dict
