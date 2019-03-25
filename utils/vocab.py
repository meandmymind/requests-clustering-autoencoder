from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import json
import os

class Vocabulary:
    """
    Maps characters to indexes and vice versa.
    Creates and stores vocab.
    """
    def __init__(self):
        self.vocab_file = os.path.dirname(os.path.abspath(__file__)) + '/vocab.json'
        self.max_input_seq_length = 50
        self.max_vocab_size = 105
        self.vocab_size = 0
        self.char2idx = None
        self.idx2char = None

    def text_to_chars(self, text):
        # string to array of characters
        text = list(text)
        text.insert(0, '<START>')

        # prune long strings
        seq_length = len(text)
        if seq_length >= self.max_input_seq_length:
            text = text[0:self.max_input_seq_length-1]
        
        text.append('<END>')
        
        return text
    
    def chars_to_indexes(self, chars):
        text_len = len(chars)
        indexes = list(range(text_len))
        for i in range(text_len):
            idx = 1 # default <UNK>
            if chars[i] in self.char2idx:
                idx = self.char2idx[chars[i]]
            indexes[i]= idx
        return indexes

    def transform_texts(self, texts):
        set_len = len(texts)
        new_texts = list(range(set_len))
        for i in range(set_len):
            text = texts[i]
            text = self.text_to_chars(text)
            text = self.chars_to_indexes(text)
            new_texts[i] = text
        
        new_texts = pad_sequences(new_texts, maxlen=self.max_input_seq_length, 
                                  padding='post', truncating='post')
        return new_texts

    def make_vocab(self, texts):
        vocab_counter = Counter()
        for text in texts:
            chars = self.text_to_chars(text)
            for char in chars:
                vocab_counter[char] += 1

        char2idx = dict()
        for idx, char_and_count in enumerate(vocab_counter.most_common(self.max_vocab_size)):
            char2idx[char_and_count[0]] = idx + 2 
        char2idx['<PAD>'] = 0
        char2idx['<UNK>'] = 1

        idx2char = dict([(idx, char) for char, idx in char2idx.items()])

        self.char2idx = char2idx
        self.idx2char = idx2char
        self.vocab_size = len(char2idx)


    def save_vocab(self, vocab_file):
        if vocab_file:
            self.vocab_file = os.path.dirname(os.path.abspath(__file__)) + '/' + vocab_file
        with open(self.vocab_file, 'w') as f:
            json.dump(self.char2idx, f)
    
    def load_vocab(self, vocab_file=None):
        if vocab_file:
            self.vocab_file = os.path.dirname(os.path.abspath(__file__)) + '/' + vocab_file
        with open(self.vocab_file, 'r') as f:
            self.char2idx = json.load(f)
            self.idx2char = dict([(idx, char) for char, idx in self.char2idx.items()])
            self.vocab_size = len(self.char2idx)
