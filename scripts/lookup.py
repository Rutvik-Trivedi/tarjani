import dataloaders
import tokenizers
import featurizers
import classifiers
import entity_extractors


lookup = {
  "cnn_classifier": classifiers.CNNClassifier.CNNClassifier,
  "lstm_classifier": classifiers.LSTMClassifier.LSTMClassifier,
  "svm_classifier": classifiers.SVMClassifier.SVMClassifier,
  "crf_loader": dataloaders.CRFLoader.CRFLoader,
  "embedding_loader": dataloaders.EmbeddingLoader.EmbeddingLoader,
  "simple_loader": dataloaders.SimpleLoader.SimpleLoader,
  "crf_classifier": entity_extractors.CRFClassifier.CRFClassifier,
  "albert_featurizer": featurizers.AlbertFeaturizer.AlbertFeaturizer,
  "embedding_featurizer": featurizers.EmbeddingFeaturizer.EmbeddingFeaturizer,
  "keras_tokenizer": tokenizers.KerasTokenizer.KerasTokenizer,
  "albert_tokenizer": tokenizers.AlbertTokenizer.AlbertTokenizer
}
