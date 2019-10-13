from tensorflow.keras.layers import Input, Dense, Embedding, Dropout, LSTM, add
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant
import embedding
import input_generator
from math import ceil
import config
import os

# image expert
img_input = Input(shape=(2048, ))
li1 = Dropout(0.2)(img_input)
li2 = Dense(256, activation='relu')(li1)

# partial caption expert
word_input = Input(shape=(1, ))

embedding_matrix = embedding.Embedder.get_instance().embedding_matrix
lw1 = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                embeddings_initializer=Constant(embedding_matrix), input_length=1, trainable=False)(word_input)

lw2 = Dropout(0.2)(lw1)
lw3 = LSTM(256)(lw2)

# merge into a single model
m1 = add([li2, lw3])
m2 = Dense(256, activation='relu')(m1)
output = Dense(embedding_matrix.shape[0], activation='softmax')(m2)

model = Model(inputs=[img_input, word_input], outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit_generator(input_generator.generator(),
                    steps_per_epoch=ceil(config.NUM_TRAIN_IMAGES/config.BATCH_SIZE),
                    epochs=30,
                    verbose=1,
                    use_multiprocessing=True)
model.save(os.path.join(config.MODEL_DIR, 'model.h5'))
