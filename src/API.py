from tensorflow.keras.models import load_model
import os
import config
import vectorizer
import data_manager
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.pyplot import imshow


img_vectors = data_manager.load_data(os.path.join(config.VECTORIZED_DATA_DIR, 'img_vectors.pkl'))
word_index = vectorizer.Description.get_instance().tokenizer.word_index
index_word = vectorizer.Description.get_instance().tokenizer.index_word
model = load_model(os.path.join(config.MODEL_DIR, 'model_1.h5'))
in_text = 'startseq'
seq = [word_index[in_text]]

while 1:
    sequence = pad_sequences([seq], maxlen=39, padding='post')[0]
    outseq = model.predict([[img_vectors['2699342860_5288e203ea.jpg']], [np.array(sequence)]], verbose=0)
    outseq = np.argmax(outseq)
    word = index_word[outseq]
    seq.append(outseq)
    if word == 'endseq':
        break
    print(word)

img = imread(os.path.join(config.IMAGE_DATA_PATH, '2699342860_5288e203ea.jpg'))
imgplot = imshow(img)
plt.show()
