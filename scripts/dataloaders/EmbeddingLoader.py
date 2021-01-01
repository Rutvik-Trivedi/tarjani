from .BaseLoader import BaseLoader


class EmbeddingLoader(BaseLoader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def name(self):
        return 'embedding_loader'

    def data(self, **kwargs):
        train_X, train_y = self._get_raw_data()
        train_X = self._remove_puncts_batch(train_X)
        train_X = self._tokenize_batch(train_X)
        train_X, train_y = self._permute_batch(train_X, train_y,
                                               permute_count=kwargs.get('permute_count', 5))
        train_X = self._remove_stopwords_batch(train_X)
        train_X = self._lemmatize_batch(train_X)
        train_X = self._join_batch(train_X)
        train_y = self._class_encode(train_y, fit=True)
        return train_X, train_y