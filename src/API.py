from tensorflow.keras.models import load_model
import os
import config
import vectorizer
import data_manager
import numpy as np


img_vectors = data_manager.load_data(os.path.join(config.VECTORIZED_DATA_DIR, 'img_vectors.pkl'))
word_index = vectorizer.Description.get_instance().tokenizer.word_index
index_word = vectorizer.Description.get_instance().tokenizer.index_word
model = load_model(os.path.join(config.MODEL_DIR, 'model.h5'))
in_text = 'startseq'

while 1:
    sequence = [word_index[in_text]]
    outseq = model.predict([[img_vectors['965444691_fe7e85bf0e.jpg']], [np.array(sequence)]], verbose=0)
    outseq = np.argmax(outseq)
    word = index_word[outseq]
    in_text = word
    print(word)
    if word == 'endseq':
        break

