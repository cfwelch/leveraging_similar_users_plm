# generates a dictionary (token to index) and an embedding layer from vectors.txt
import pickle
import numpy as np

def main():
    # vectors.txt has already included '<unk>' as the last token
    vocabulary_size = 55455
    embedding_size = 200
    token_to_index = {}
    embedding_layer = np.zeros((vocabulary_size, embedding_size), dtype=float)

    with open('vectors.txt', 'r') as file:
        # There are 201 elements in each line separated with whitespaces
        # The 0th element is a token
        # and the remaining elements is the corresponding vector.
        for index, line in enumerate(file):
            # index starts with 0
            info = line.split(' ')
            token_to_index[info[0]] = index
            embedding_layer[index] = [float(entry) for entry in info[1: ]]

    with open('GloVe_200_dictionary.pkl', 'wb') as file:
        pickle.dump(token_to_index, file)

    with open('GloVe_200_embedding_layer.pkl', 'wb') as file:
        pickle.dump(embedding_layer, file)


if __name__ == '__main__':
    main()
