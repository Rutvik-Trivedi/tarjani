import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import backend as K


class SiameseNet(tf.keras.layers.Layer):

    def __init__(self, model):
        self.model = model
        super().__init__()

    def call(self, feat):
        feats = self.model(feat[0])
        pfeats = self.model(feat[1])
        final = tf.stack([feats, pfeats])
        return tf.transpose(final, perm=(1,2,0))

class Distance(tf.keras.layers.Layer):

    def __init__(self, margin):
        self.margin = margin
        super().__init__()

    def call(self, final):
        base_loss = K.sum(K.square(final[:,:,0] - final[:,:,1]), axis=1)
        return base_loss



class ModelCreator():

    def __init__(self, feature_extractor='../../model/vision/face-rec/face_rec.h5'):
        try:
            self.feature_extractor = tf.keras.models.load_model(feature_extractor, compile=False)
        except:
            print("Face Recognition model not found. Please follow the installation guide to download and install the model")

    def get_model(self, margin=1):
        image = tf.keras.layers.Input(shape=(299,299,3), name='image_input')
        positive = tf.keras.layers.Input(shape=(299,299,3), name='positive_input')
        siamese = SiameseNet(self.feature_extractor)([image, positive])
        distance = Distance(margin)(siamese)
        model = tf.keras.models.Model(inputs = [image, positive], outputs=distance)
        return model, self.feature_extractor
