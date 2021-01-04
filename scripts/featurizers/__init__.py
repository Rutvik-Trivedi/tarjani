import importlib
import os


modules = os.listdir('featurizers')
for i in modules:
    if i.endswith('.py'):
        name = 'featurizers.'+i.strip('.py')
        importlib.import_module(name)
