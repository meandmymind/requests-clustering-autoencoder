import pandas as pd
import numpy as np
import requests
import os

from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input, Embedding
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K

from utils.vocab import Vocabulary

from sklearn.cluster import KMeans
from joblib import load


# Utils for working with API 
def download_sample():
    download_url = ""

    try:
        resp = requests.get(download_url)
        sample = resp.json()

        # Sort out invalid samples like {'error': 'You already downloaded this vector'}
        if set(sample.keys()) != set(valid_keys):
            failed["num"] += 1
            failed["responses"].append(resp.text)
            print("Invalid sample")
            print(failed)
            return None

    # Sometimes decoded samples are not valid json objects
    except ValueError:
        failed["num"] += 1
        try:
            failed["responses"].append(resp)
            print("Invalid response")
        # If download failed, resp is undound
        except UnboundLocalError:
            print("Bad URL")
            
        return None

    return sample

def upload_sample(classified_sample):
    upload_url = ""

    try:
        resp = requests.post(upload_url, classified_sample)
        return resp.text
    except:
        print("Download failed")
        return None

# Autoencoder model
class AutoEncoder(object):
    """ 4-layer LSTM Autoencoder
    Parameters
        vocab: Vocabulary object that stores char-to-int mappings 
            and text processing functions.
        hidden_size: Dimension of latent space between encoder and decoder.
            Default: 64.
        dropout: Fraction of units to be randomly ignored during training.
            Both linear and recurrent dropout are used.
            Default: 0.2.
    """
    DEFAULT_HIDDEN_UNITS = 64
    DEFAULT_DROPOUT = 0.2
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_EPOCHS = 100
    DEFAULT_MODEL_WEIGHT_PATH = './models/autoencoder-weights.h5'
    model_name = 'autoencoder'

    def __init__(self, vocab, hidden_size=None, dropout=None):
        if not hidden_size:
            self.hidden_size = AutoEncoder.DEFAULT_HIDDEN_UNITS
        if not dropout:
            self.dropout = AutoEncoder.DEFAULT_DROPOUT
            
        self.vocab = vocab
        self.vocab_size = vocab.vocab_size
        self.max_input_seq_length = vocab.max_input_seq_length

        encoder_inputs = Input(shape=(None,), name='encoder_inputs')
        encoder_embedding = Embedding(input_dim=self.vocab_size,
                                      output_dim=self.hidden_size,
                                      input_length=self.max_input_seq_length,
                                      name='encoder_embedding')
        encoder_lstm = LSTM(units=self.hidden_size,
                            return_state=True,
                            dropout=self.dropout,
                            recurrent_dropout=self.dropout,
                            name='encoder_lstm')
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding(encoder_inputs))
        encoder_states = [encoder_state_h, encoder_state_c]

        decoder_inputs = Input(shape=(None, self.vocab_size), name='decoder_inputs')
        decoder_lstm = LSTM(units=self.hidden_size, return_state=True,
                            return_sequences=True, name='decoder_lstm')
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                         initial_state=encoder_states)
        decoder_dense = Dense(units=self.vocab_size, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        self.model = model

        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_inputs = [Input(shape=(self.hidden_size,)), Input(shape=(self.hidden_size,))]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs,
                                                         initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_state_inputs,
                                   [decoder_outputs] + decoder_states)

    def load_weights(self, weight_file_path):
        if os.path.exists(weight_file_path):
            self.model.load_weights(weight_file_path)

    def generate_batch(self, samples, batch_size):
        num_batches = len(samples) // batch_size
        while True:
            for batchIdx in range(0, num_batches):
                start = batchIdx * batch_size
                end = (batchIdx + 1) * batch_size
                encoder_input_data_batch = pad_sequences(samples[start:end], self.max_input_seq_length)
                decoder_target_data_batch = np.zeros(shape=(batch_size, self.max_input_seq_length,
                                                            self.vocab_size))
                decoder_input_data_batch = np.zeros(shape=(batch_size, self.max_input_seq_length,
                                                           self.vocab_size))
                for textIdx, chars in enumerate(samples[start:end]):
                    for idx, char in enumerate(chars):
                        if char != 1: # not <UNK>
                            decoder_input_data_batch[textIdx, idx, char] = 1
                            if idx > 0:
                                decoder_target_data_batch[textIdx, idx - 1, char] = 1
                                
                yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch

    def fit(self, Xtrain, Xtest, epochs=None, batch_size=None, model_dir_path=None):
        """
        Train model.
        Parameters:
            Xtrain: Array of strings for training.
            Xtest: Array of strings for validation.
            epochs: Number of full forward passes of all training samples.
                Default: 50.
            batch_size: Number of samples to pass for one iteration.
                Default: 64.
            model_dir_path: File path with extension '.h5'
                for saving weights of the network. (https://www.h5py.org/)
                Default: ./models/autoencoder-weights.h5
        """

        if not epochs:
            epochs = AutoEncoder.DEFAULT_EPOCHS
        if not model_dir_path:
            model_dir_path = './models'
        if not batch_size:
            batch_size = AutoEncoder.DEFAULT_BATCH_SIZE
        if not weight_file_path:
            weight_file_path = AutoEncoder.DEFAULT_WEIGHT_FILE_PATH

        checkpoint = ModelCheckpoint(weight_file_path)

        Xtrain = self.vocab.transform_texts(Xtrain)
        Xtest = self.vocab.transform_texts(Xtest)

        train_gen = self.generate_batch(Xtrain, batch_size)
        test_gen = self.generate_batch(Xtest, batch_size)

        train_num_batches = len(Xtrain) // batch_size
        test_num_batches = len(Xtest) // batch_size

        history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                           epochs=epochs, verbose=True,
                                           validation_data=test_gen, validation_steps=test_num_batches,
                                           callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)
        
        return history

    def predict(self, input_text):
        """
        Process input texts and reconstruct them.
        Used to check how well model "understands" input texts.
        Parameters:
             input_text: 
        """
        input_seq = self.vocab.transform_texts([input_text])
        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, self.vocab_size))
        target_seq[0, 0, self.vocab.char2idx['<START>']] = 1
        target_text = ''
        target_text_len = 0
        terminated = False
        while not terminated:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            sample_token_idx = np.argmax(output_tokens[0, -1, :])
            sample_char = self.vocab.idx2char[sample_token_idx]
            target_text_len += 1

            if sample_char != '<START>' and sample_char != '<END>':
                target_text += ' ' + sample_char

            if sample_char == '<END>' or target_text_len >= self.max_input_seq_length:
                terminated = True

            target_seq = np.zeros((1, 1, self.vocab_size))
            target_seq[0, 0, sample_token_idx] = 1

            states_value = [h, c]

        return target_text.strip()   

# Text processing
def get_features(vector):
    # vector: string -> vector_embedding: [char_indices]
    # [char_indices] are padded with 0 to the right
    vector_embedding = vocab.transform_texts([vector])
    
    # vector_embedding: [char_indices] -> encoders_states: [state_h, state_c]
    encoder_states = encoder.predict([vector_embedding])
    
    # use state_h as feature
    vector_features = encoder_states[0]
    
    return vector_features

# Prediction
def predict_class(vector_features):
    vector_class = kmeans.predict(vector_features)[0]
    # "0"-class is invalid
    if vector_class == 0:
        vector_class = 25
    return vector_class

# Main code

os.environ["CUDA_VISIBLE_DEVICES"] = ''

vocab = Vocabulary()
vocab.load_vocab('vocab.json')
autoencoder = AutoEncoder(vocab)
autoencoder.load_weights('./models/autoencoder-weights.h5')
encoder = autoencoder.encoder_model
kmeans = load('./models/kmeans-weights.joblib')


# make valid template
valid_keys = ("id", "vector")
samples = pd.DataFrame(columns=valid_keys)
failed = {"num": 0, "responses": []}
num_samples = 1

for i in range(num_samples):
    print(i, end=' ')
    sample = download_sample()
    # if sample is invalid, continue to next one 
    if sample == None:
        continue
    # make prediction
    vector = sample['vector']
    vector_features = get_features(vector)
    vector_class = predict_class(vector_features)
    # upload result
    sample['class'] = vector_class
    classified_vector = {"vector": sample["id"], "class": sample["class"]}
    upload_sample(classified_vector)
    # save sample for further training
    samples = samples.append([sample], sort=True)

print("Finished!")

samples = samples.reset_index(drop=True)
samples.to_csv('./vectors/samples-script.csv', index=False)