import pandas as pd
import requests
import os

from src.utils.vocab import Vocabulary
from src.autoencoder import AutoEncoder

from sklearn.cluster import KMeans
from joblib import load


# Configure app
os.environ["CUDA_VISIBLE_DEVICES"] = ''
vocab = Vocabulary()
vocab.load_vocab('vocab.json')
autoencoder = AutoEncoder(vocab)
autoencoder.load_weights(os.path.dirname(os.path.abspath(__file__)) + '/models/autoencoder-weights.h5')
encoder = autoencoder.encoder_model
kmeans = load(os.path.dirname(os.path.abspath(__file__)) + '/models/kmeans-weights.joblib')

# make valid template
valid_keys = ("id", "vector")
samples = pd.DataFrame(columns=valid_keys)
failed = {"num": 0, "responses": []}
num_samples = 1


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


# Text processing
def get_features(vector):
    # vector: string -> vector_embedding: [char_indices]
    # [char_indices] are padded with 0 to the right
    vector_embedding = vocab.transform_texts([vector])
    
    print(f"\nvector is {vector}\nembedding is {vector_embedding}")
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


def predict(sample):
    # make prediction
    vector = sample['vector']
    vector_features = get_features(vector)
    vector_class = predict_class(vector_features)

    # return result
    sample['class'] = vector_class
    classified_vector = {"vector": sample["id"], "class": sample["class"]}
    return classified_vector

def download_and_predict():
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
    samples.to_csv('./vectors/samples-script-something.csv', index=False)

