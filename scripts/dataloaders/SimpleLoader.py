from .BaseLoader import BaseLoader


class SimpleLoader(BaseLoader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = None
        self.requires_save = False


    def name(self):
        return 'simple_loader'

    def data(self, **kwargs):
        train_X, train_y = self._get_raw_data()
        train_y = self._class_encode(train_y, fit=True)
        return train_X, train_y
