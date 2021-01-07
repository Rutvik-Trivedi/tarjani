from .BaseLoader import BaseLoader


class EmbeddingLoader(BaseLoader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = None
        self.requires_save = False


    def name(self):
        return 'embedding_loader'

    def __call__(self, **kwargs):
        train_X, train_y = self._get_raw_data()
        train_X = self._remove_puncts_batch(train_X)
        train_X = self._tokenize_batch(train_X)
        train_X, train_y = self._permute_batch(train_X, train_y,
                                               permute_count=kwargs.get('permute_count', 5))
        train_X = self._remove_stopwords_batch(train_X)
        train_X = self._lemmatize_batch(train_X)
        train_X = self._join_batch(train_X)
        train_y = self._class_encode(train_y, fit=True)
        self._set_num_classes(train_y)
        return train_X, train_y

    def prepare_query(self, sentence):
        sentence = [sentence]
        sentence = self._remove_puncts_batch(sentence)
        sentence = self._tokenize_batch(sentence)
        sentence = self._remove_stopwords_batch(sentence)
        sentence = self._lemmatize_batch(sentence)
        sentence = self._join_batch(sentence)
        return sentence
