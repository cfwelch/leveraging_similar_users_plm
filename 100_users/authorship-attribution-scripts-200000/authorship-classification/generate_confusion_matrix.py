import IPython, torch, time, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.vocab import Vectors


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=True,
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, text, text_lengths):
        # text: sent_len x batch_size
        embedded = self.dropout(self.embedding(text))
        # embedded: sent_len x batch_size x emb_dim
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        # output: sent_len x batch_size x (hid_dim * num_directions)
        # hidden: (num_layers * num_directions) x batch_size x hid_dim
        # cell:   (num_layers * num_directions) x batch_size x hid_dim
        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers and apply dropout
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        # hidden: batch_size x (hid_dim * num_directions)
        return self.fc(hidden)


def multiclass_accuracy(predictions, labels):
    return torch.sum(labels == torch.argmax(predictions, dim=1)) * 1.0 / labels.shape[0]


def generate_confusion_matrix(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    confusion_matrix = torch.zeros((100, 100), dtype=torch.int)
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.post
            predictions = model(text, text_lengths)
            loss = criterion(predictions, batch.author)
            acc = multiclass_accuracy(predictions, batch.author)
            predicted = torch.argmax(predictions, dim=1)
            for i in range(len(batch.author)):
                confusion_matrix[batch.author[i]][predicted[i]] += 1
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator), confusion_matrix


TEXT = Field(sequential=True, include_lengths=True)
LABEL = Field(sequential=False, use_vocab=False)

train_datafields = [("post", TEXT), ("author", LABEL)]
train_dataset = TabularDataset(
           path="../../data/anchor_posts_and_labels_training.csv",
           format='csv',
           csv_reader_params={'delimiter':'\t', 'quotechar':'"'},
           skip_header=True,
           fields=train_datafields)

valid_datafields = [("post", TEXT), ("author", LABEL)]
valid_dataset = TabularDataset(
          path="../../data/anchor_posts_and_labels_validation.csv",
          format='csv',
          csv_reader_params={'delimiter':'\t', 'quotechar':'"'},
          skip_header=True,
          fields=valid_datafields)

test_datafields = [("post", TEXT), ("author", LABEL)]
test_dataset = TabularDataset(
           path="../../data/anchor_posts_and_labels_test.csv",
           format='csv',
           csv_reader_params={'delimiter':'\t', 'quotechar':'"'},
           skip_header=True,
           fields=test_datafields)

# TEXT.build_vocab(train_dataset, valid_dataset, test_dataset,
#     min_freq = 3,
#     vectors = "glove.6B.200d",
#     unk_init = torch.Tensor.normal_)

vectors = Vectors(name="glove.6B.200d.txt", cache='./')
TEXT.build_vocab(train_dataset, valid_dataset, test_dataset,
    min_freq = 3,
    vectors = vectors)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device)

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_dataset, valid_dataset, test_dataset),
    batch_size = 64,
    sort_key=lambda x: len(x.post),
    sort_within_batch = True,
    device = device)

VOCAB_SIZE = len(TEXT.vocab)
EMBEDDING_DIM = 200
HIDDEN_DIM = 400
OUTPUT_DIM = 100
N_LAYERS = 3
DROPOUT = 0.5
LEARNING_RATE = 1e-3
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model = RNN(VOCAB_SIZE,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            OUTPUT_DIM,
            N_LAYERS,
            DROPOUT,
            PAD_IDX)

model.load_state_dict(torch.load('best_model_1e-3.pt'))
criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)

test_loss, test_acc, confusion_matrix = generate_confusion_matrix(model, test_iterator, criterion)
print(f'\t Val. Loss: {test_loss:.3f} |  Val. Acc: {test_acc*100:.2f}%')

with open('confusion_matrix.pkl', 'wb') as file:
    pickle.dump(confusion_matrix.cpu().numpy(), file)

# IPython.embed()
