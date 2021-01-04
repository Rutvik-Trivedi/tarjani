import importlib
import os


modules = os.listdir('classifiers')
for i in modules:
    if i.endswith('.py'):
        name = 'classifiers.'+i.strip('.py')
        importlib.import_module(name)
