import IPython, torch, time, argparse, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.vocab import Vectors


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers,
                 dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=num_layers,
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


def generate_confusion_matrix(model, iterator):
    confusion_matrix = torch.zeros((10000, 10000), dtype=torch.int)
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.post
            predictions = model(text, text_lengths)
            predicted = torch.argmax(predictions, dim=1)
            for i in range(len(batch.author)):
                confusion_matrix[batch.author[i]][predicted[i]] += 1
    return confusion_matrix



parser = argparse.ArgumentParser(description='Authorship Classification Model')
parser.add_argument('--min_freq', type=int, default=15,
                    help='min frequency to add to vocabulary')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size for training, validation and test')
parser.add_argument('--hidden_dim', type=int, default=400,
                    help='hidden dimension for LSTM')
parser.add_argument('--num_layers', type=int, default=3,
                    help='layers of LSTM')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout probability')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--pretrained_model', type=str, required=True,
                    help='path of the pretrained model')

args = parser.parse_args()
print('Args:', args)

TEXT = Field(sequential=True, include_lengths=True)
LABEL = Field(sequential=False, use_vocab=False)

train_datafields = [("post", TEXT), ("author", LABEL)]
train_dataset = TabularDataset(
           path="../../data_10000/anchor_posts_and_labels_training.csv",
           format='csv',
           csv_reader_params={'delimiter':'\t', 'quotechar':'"'},
           skip_header=True,
           fields=train_datafields)

valid_datafields = [("post", TEXT), ("author", LABEL)]
valid_dataset = TabularDataset(
          path="../../data_10000/anchor_posts_and_labels_validation.csv",
          format='csv',
          csv_reader_params={'delimiter':'\t', 'quotechar':'"'},
          skip_header=True,
          fields=valid_datafields)

test_datafields = [("post", TEXT), ("author", LABEL)]
test_dataset = TabularDataset(
           path="../../data_10000/anchor_posts_and_labels_test.csv",
           format='csv',
           csv_reader_params={'delimiter':'\t', 'quotechar':'"'},
           skip_header=True,
           fields=test_datafields)

vectors = Vectors(name="glove.6B.200d.txt", cache='./')
TEXT.build_vocab(train_dataset, valid_dataset, test_dataset,
    min_freq = args.min_freq,
    vectors = vectors)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device)

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_dataset, valid_dataset, test_dataset),
    batch_size = args.batch_size,
    sort_key=lambda x: len(x.post),
    sort_within_batch = True,
    device = device)

pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
unk_idx = TEXT.vocab.stoi[TEXT.unk_token]

embedding_dim = 200
output_dim = 10000
model = RNN(len(TEXT.vocab),
            embedding_dim,
            args.hidden_dim,
            output_dim,
            args.num_layers,
            args.dropout,
            pad_idx)
model.load_state_dict(torch.load(args.pretrained_model))
model = model.to(device)

confusion_matrix = generate_confusion_matrix(model, test_iterator)
confusion_matrix = confusion_matrix.cpu().numpy()
normalized_confusion_matrix = confusion_matrix / np.sum(confusion_matrix, axis=1, keepdims=True)
similarity_matrix = (normalized_confusion_matrix + normalized_confusion_matrix.T) / 2

with open('anchor_similarity_matrix.pkl', 'wb') as file:
    pickle.dump(similarity_matrix, file)

# IPython.embed()
