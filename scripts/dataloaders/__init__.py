import importlib
import os


modules = os.listdir('dataloaders')
for i in modules:
    if i.endswith('.py'):
        name = 'dataloaders.'+i.strip('.py')
        importlib.import_module(name)
