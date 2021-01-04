import importlib
import os


modules = os.listdir('tokenizers')
for i in modules:
    if i.endswith('.py'):
        name = 'tokenizers.'+i.strip('.py')
        importlib.import_module(name)
