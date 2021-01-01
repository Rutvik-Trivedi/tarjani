from dataloaders import SimpleLoader, EmbeddingLoader, CRFLoader
from tokenizers import KerasTokenizer
from featurizers import AlbertFeaturizer, EmbeddingFeaturizer
from classifiers import (
    CNNClassifier, SVCClassifier, LSTMClassifier
)
from entity_extractors import CRFClassifier


lookup = {
  "cnn_classifier": CNNClassifier.CNNClassifier,
  "lstm_classifier": LSTMClassifier.LSTMClassifier,
  "svm_classifier": SVCClassifier.SVCClassifier,
  "crf_loader": CRFLoader.CRFLoader,
  "embedding_loader": EmbeddingLoader.EmbeddingLoader,
  "simple_loader": SimpleLoader.SimpleLoader,
  "crf_classifier": CRFClassifier.CRFClassifier,
  "albert_featurizer": AlbertFeaturizer.AlbertFeaturizer,
  "embedding_featurizer": EmbeddingFeaturizer.EmbeddingFeaturizer,
  "keras_tokenizer": KerasTokenizer.KerasTokenizer
}
