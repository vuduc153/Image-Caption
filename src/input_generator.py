import vectorizer
import config
import data_manager
import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


def generator():

    img_vectors = data_manager.load_data(os.path.join(config.VECTORIZED_DATA_DIR, 'img_vectors.pkl'))
    description_handler = vectorizer.Description.get_instance()
    max_length = description_handler.max_length
    num_word = min(config.MAX_NUM_WORDS, len(description_handler.tokenizer.word_index) + 1)

    with open(config.TRAIN_DATA_PATH) as f:

        training_images = list()

        for line in f:
            # remove newline character
            line = line[0:-1]
            training_images.append(line)

        num_image = 0
        input_1, input_2, output = list(), list(), list()

        while 1:
            num_image += 1
            document_vectors = description_handler.img_desc[training_images[num_image-1]]
            img_vector = img_vectors[training_images[num_image-1]]
            for vector in document_vectors:
                for i in range(1, len(vector)):
                    input_1.append(img_vector)
                    input_2.append(pad_sequences([vector[:i]], maxlen=max_length, padding='post')[0])
                    output.append(to_categorical(vector[i], num_classes=num_word))

            if num_image % config.BATCH_SIZE == 0 or num_image == config.NUM_TRAIN_IMAGES:
                yield ([np.array(input_1), np.array(input_2)], np.array(output))
                input_1, input_2, output = list(), list(), list()
                if num_image == config.NUM_TRAIN_IMAGES:
                    num_image = 0


if __name__ == '__main__':
    a = generator()
    next(a)
