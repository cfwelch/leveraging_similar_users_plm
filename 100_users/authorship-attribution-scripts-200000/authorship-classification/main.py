import IPython, torch, time
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

pretrained_embeddings = TEXT.vocab.vectors
# print(pretrained_embeddings.shape)
model.embedding.weight.data.copy_(pretrained_embeddings)

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)
"""
NUM_EPOCHS = 100
best_valid_loss = float('inf')
epoch_no_improvement = 0
for epoch in range(NUM_EPOCHS):
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
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        epoch_no_improvement += 1
    if epoch_no_improvement >= 10:
        break
"""
model.load_state_dict(torch.load('best_model_1e-3.pt'))
test_loss, test_acc = evaluate(model, test_iterator, criterion)
print(f'\tTest Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
# IPython.embed()
