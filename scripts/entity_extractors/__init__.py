import importlib
import os


modules = os.listdir('entity_extractors')
for i in modules:
    if i.endswith('.py'):
        name = 'entity_extractors.'+i.strip('.py')
        importlib.import_module(name)
