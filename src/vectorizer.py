import os
import config
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def image_vectorize():

    model = InceptionV3(weights='imagenet')
    model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

    img_vectors = dict()

    for idx, file in enumerate(os.listdir(config.IMAGE_DATA_PATH)):

        # load image
        filepath = os.path.join(config.IMAGE_DATA_PATH, file)
        img = load_img(path=filepath, target_size=(299, 299))
        # convert PIL type to ndarray
        img_input = img_to_array(img)
        # append sample dimension
        img_input = np.expand_dims(img_input, axis=0)
        # model preprocess
        img_input = preprocess_input(img_input)
        # get output
        output_vector = model.predict(img_input).flatten()

        img_vectors[file] = output_vector

        print("Vectorized image " + str(idx) + "\n")

    return img_vectors


class Description:

    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if Description.__instance is None:
            Description()
        return Description.__instance

    def __init__(self):

        if Description.__instance is not None:
            print('Description class is a singleton!')
        else:
            Description.__instance = self
            self.img_desc = dict()
            self.desc = list()
            self.max_length = 0
            self.tokenizer = self.__tokenize()

    def __read_file(self):

        with open(config.DESC_DATA_PATH, 'r') as f:
            for line in f:
                values = line.split('#')
                filename = values[0]
                caption = values[1].split('\t', 1)
                idx = caption[0]

                if idx == '0':
                    self.img_desc[filename] = list()

                desc = "startseq " + caption[1] + " endseq"
                self.img_desc[filename].append(desc)
                self.desc.append(desc)

    def __tokenize(self):

        self.__read_file()
        tokenizer = Tokenizer(num_words=config.MAX_NUM_WORDS)
        tokenizer.fit_on_texts(self.desc)
        for file, desc in self.img_desc.items():
            self.img_desc[file] = tokenizer.texts_to_sequences(desc)
            # length of the longest description for this picture
            maxlen = len(max(self.img_desc[file], key=len))
            if maxlen > self.max_length:
                self.max_length = maxlen
        return tokenizer


if __name__ == '__main__':

    description_handler = Description.get_instance()
    print(description_handler.max_length)


