class SettingsUpdater(object):

    def __init__(self, obj, settings, type):
        self.obj = obj
        self.sets = settings[type]

    def __enter__(self):
        self.sets.update(self.obj.__dict__)

    def __exit__(self, exception_type, exception_value, traceback):
        self.sets.update(self.obj.__dict__)
