import pickle


def store_data(vector_list, path):

    with open(path, 'wb') as f:
        pickle.dump(vector_list, f)


def load_data(path):

    with open(path, 'rb') as f:
        return pickle.load(f)
