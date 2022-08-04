import IPython, torch, time, argparse
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


def multiclass_accuracy(predictions, labels):
    return torch.sum(labels == torch.argmax(predictions, dim=1)) * 1.0 / labels.shape[0]


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.post
        predictions = model(text, text_lengths)
        loss = criterion(predictions, batch.author)
        acc = multiclass_accuracy(predictions, batch.author)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.post
            predictions = model(text, text_lengths)
            loss = criterion(predictions, batch.author)
            acc = multiclass_accuracy(predictions, batch.author)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)



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
parser.add_argument('--num_epochs', type=int, default=100,
                    help='num training epochs')
parser.add_argument('--save', type=str, required=True,
                    help='path to save the trained model')

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

pretrained_embeddings = TEXT.vocab.vectors
# print(pretrained_embeddings.shape)
model.embedding.weight.data.copy_(pretrained_embeddings)

model.embedding.weight.data[unk_idx] = torch.zeros(embedding_dim)
model.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)


best_valid_loss = float('inf')
epoch_no_improvement = 0
for epoch in range(args.num_epochs):
    start_time = time.time()
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    end_time = time.time()
    print('epoch: {} time: {:.2f}'.format(epoch, end_time-start_time))

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    if valid_loss < best_valid_loss:
        epoch_no_improvement = 0
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), args.save)
    else:
        epoch_no_improvement += 1
    if epoch_no_improvement >= 10:
        break

model.load_state_dict(torch.load(args.save))
test_loss, test_acc = evaluate(model, test_iterator, criterion)
print(f'\tTest Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
