'''
This is an experimental feature. TARJANI is aimed to have vision abilities as well.
It will be able to understand images while in interactive mode. Please provide your
feedback on this feature by using the submit_feedback.py script or by
visiting http://tarjani.is-great.net

Usage:

BASH:
python3 vision.py --path [PATH TO IMAGE]

API:
from vision import understand_image
print(umderstand_image(image = 'path/to/image'))
'''

from argparse import ArgumentParser
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import pickle

def info():
    print('''
    This is an experimental feature. TARJANI is aimed to have vision abilities as well.
    It will be able to understand images while in interactive mode. Please provide your
    feedback on this feature by using the submit_feedback.py script or by
    visiting http://tarjani.is-great.net
    ''')

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--path', '-p', help="Path of the image file", type=str, default='NONE')
    return parser

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

def get_tokenizer():
    with open('../../model/vision/image-captioning/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

def get_max_length():
    with open('../../model/vision/image-captioning/max_length.pkl', 'rb') as f:
        max_length = int(pickle.load(f))
    return max_length


class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    hidden_with_time_axis = tf.expand_dims(hidden, 1)
    attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                         self.W2(hidden_with_time_axis)))
    score = self.V(attention_hidden_layer)
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    context_vector, attention_weights = self.attention(features, hidden)
    x = self.embedding(x)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    output, state = self.gru(x)
    x = self.fc1(output)
    x = tf.reshape(x, (-1, x.shape[2]))
    x = self.fc2(x)
    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))


def initialize():
    ## Initializations
    opts = {}
    opts['weights'] = '../../model/vision/image-captioning/inception.h5'
    opts['max_length'] = get_max_length()
    opts['tokenizer'] = get_tokenizer()
    opts['image_model'] = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights=opts['weights'])
    opts['new_input'] = opts['image_model'].input
    opts['hidden_layer'] = opts['image_model'].layers[-1].output
    opts['image_features_extract_model'] = tf.keras.Model(opts['new_input'], opts['hidden_layer'])
    opts['top_k'] = 5000
    opts['embedding_dim'] = 256
    opts['units'] = 512
    opts['vocab_size'] = opts['top_k'] + 1
    opts['features_shape'] = 2048
    opts['attention_features_shape'] = 64
    opts['encoder'] = CNN_Encoder(opts['embedding_dim'])
    opts['decoder'] = RNN_Decoder(opts['embedding_dim'], opts['units'], opts['vocab_size'])
    opts['optimizer'] = tf.keras.optimizers.Adam()
    opts['checkpoint_path'] = "../../model/vision/image-captioning/checkpoint"
    opts['ckpt'] = tf.train.Checkpoint(encoder=opts['encoder'],
                               decoder=opts['decoder'],
                               optimizer = opts['optimizer'])
    opts['ckpt_manager'] = tf.train.CheckpointManager(opts['ckpt'], opts['checkpoint_path'], max_to_keep=5)
    if opts['ckpt_manager'].latest_checkpoint:
      opts['ckpt'].restore(opts['ckpt_manager'].latest_checkpoint).expect_partial()
    return opts





def evaluate(image):
    opts = initialize()
    attention_plot = np.zeros((opts['max_length'], opts['attention_features_shape']))

    hidden = opts['decoder'].reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = opts['image_features_extract_model'](temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = opts['encoder'](img_tensor_val)

    dec_input = tf.expand_dims([opts['tokenizer'].word_index['<start>']], 0)
    result = []

    for i in range(opts['max_length']):
        predictions, hidden, attention_weights = opts['decoder'](dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(opts['tokenizer'].index_word[predicted_id])

        if opts['tokenizer'].index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot




def understand_image(image=None):
    parser = get_parser()
    args = parser.parse_args()
    if args.path == 'NONE':
        args.path = image
    result, _ = evaluate(args.path)
    return 'This is a ' + ' '.join(result[:-1])

if __name__ == '__main__':
    result = understand_image()
    print('Predicted Caption: {}'.format(result))
