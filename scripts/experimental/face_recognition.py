'''
This is an experimental feature. TARJANI is aimed to have vision abilities as well.
It will be able to recognize faces in images while in interactive mode.
Please provide your feedback on this feature by using the submit_feedback.py
script or by visiting http://tarjani.is-great.net

Usage:

API:
from face_recognition import Recognizer
recognizer = Recognizer()
image_to_recognize = 'path/to/image.jpg'
print(recognizer.evaluate(image_to_recognize))

For complete documentation, see the documentation file
'''


import warnings
warnings.filterwarnings('ignore')
from model import ModelCreator as Creator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import cv2

class Recognizer():

    def __init__(self, feature_extractor = '../../model/vision/face-rec/face_rec.h5', cascade_classifier='../../model/vision/face-rec/haarcascade_frontalface_default.xml'):
        model_creator = Creator()
        self.model, self.feature_extractor = model_creator.get_model(feature_extractor)
        self.face_cascade = cv2.CascadeClassifier(cascade_classifier)
        if not os.path.exists('data/'):
            os.mkdir('data')

    def info(self):
        print('''
        This is an experimental feature. TARJANI is aimed to have vision abilities as well.
        It will be able to recognize faces in images while in interactive mode.
        Please provide your feedback on this feature by using the submit_feedback.py
        script or by visiting http://tarjani.is-great.net

        Usage:

        API:
        from face_recognition import Recognizer
        recognizer = Recognizer()
        image_to_recognize = 'path/to/image.jpg'
        print(recognizer.evaluate(image_to_recognize))

        For complete documentation, see the documentation file
        ''')

    def preprocess_single_image(self, positive, dir, suppress_error=True):
        img = tf.io.read_file(dir + positive)
        img = tf.image.decode_jpeg(img, channels=3)
        img = np.asarray(img)
        try:
            (x, y, h, w) = self.face_cascade.detectMultiScale(img, 1.3, 5)[0]
        except:
            if suppress_error:
                raise ValueError('The image used should have exactly one detectable frontal face for better recognition. Please check the input image {}'.format(dir + positive))
            else:
                pass
        img = img[y:y+h, x:x+w]
        img = tf.image.resize(img, (299, 299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return np.array(img, ndmin=4)

    def preprocess_input_image(self, image, num_classes):
        img = [self.preprocess_single_image(image, dir='')] * num_classes
        return np.concatenate(img)

    def get_class_images(self, dir):
        pos = []
        l = os.listdir(dir)
        for i in l:
            pos.append(np.load(dir+i))
        return np.concatenate(pos), l


    def add_image(self, image, name, dir='data/', delete_original=True):
        img = self.preprocess_single_image(image, dir='', suppress_error=False)
        encoded_image = self.feature_extractor(img)
        np.save(dir+name, encoded_image)
        if delete_original:
            os.remove(image)
        print("Image Added")
        return 1

    def remove_image(self, name, dir='data/'):
        try:
            os.remove(dir+image+'.npy')
            print("Removed Image")
            return 1
        except:
            print("No image found")
            return 0

    def evaluate(self, image, threshold=0.18):
        positives, classes = self.get_class_images(dir='data/')
        num_classes = len(classes)
        images = self.preprocess_input_image(image, num_classes)
        result = self.model([images, positives])
        min_ = K.min(result)
        if min <= threshold:
            return classes[K.argmin(result)].split('.')[0]
        else:
            return 'No match found'
