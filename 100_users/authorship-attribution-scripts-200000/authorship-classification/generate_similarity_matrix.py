import pickle, IPython
import numpy as np

with open('confusion_matrix.pkl', 'rb') as file:
    confusion_matrix = pickle.load(file)

normalized_confusion_matrix = confusion_matrix / np.sum(confusion_matrix, axis=1, keepdims=True)
similarity_matrix = (normalized_confusion_matrix + normalized_confusion_matrix.T) / 2

with open('similarity_matrix.pkl', 'wb') as file:
    pickle.dump(similarity_matrix, file)

# IPython.embed()
