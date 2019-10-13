import config
import numpy as np
import vectorizer
import os


class Embedder:

    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if Embedder.__instance is None:
            Embedder()
        return Embedder.__instance

    def __init__(self):

        if Embedder.__instance is not None:
            print('Embedder class is a singleton!')
        else:
            Embedder.__instance = self
            self.embedding = dict()
            self.embedding_matrix = None
            self.__load_embedding()
            self.__build_embedding_matrix()

    def __load_embedding(self):

        with open(os.path.join(config.GLOVE_DIR, 'glove.6B.200d.txt')) as f:
            for line in f:
                word, vector = line.split(maxsplit=1)
                self.embedding[word] = np.fromstring(vector, 'f', sep=' ')

    def __build_embedding_matrix(self):

        description_handler = vectorizer.Description.get_instance()
        word_index = description_handler.tokenizer.word_index

        num_word = min(config.MAX_NUM_WORDS, len(word_index) + 1)
        self.embedding_matrix = np.zeros((num_word, config.EMBEDDING_DIM))
        for word, idx in word_index.items():
            if idx >= config.MAX_NUM_WORDS:
                continue
            embedding_vector = self.embedding.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[idx] = embedding_vector

