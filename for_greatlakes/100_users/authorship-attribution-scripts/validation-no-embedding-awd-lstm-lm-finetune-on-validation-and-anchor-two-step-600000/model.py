import torch, pickle
import torch.nn as nn

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout

class RNNModel(nn.Module):
    """Container module with a token encoder, a recurrent module, and a token decoder."""

    def __init__(self, pretrained_model, rnn_type, pretrained_token_embedding, freeze_parameters, ntoken, token_emsize, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False):
        super(RNNModel, self).__init__()
        self.rnn_type = rnn_type
        # pretrained_token_embedding should be a file path
        self.pretrained_token_embedding = pretrained_token_embedding
        self.ntoken = ntoken
        self.token_emsize = token_emsize
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouth = dropouth
        self.dropouti = dropouti
        self.dropoute = dropoute
        self.tie_weights = tie_weights

        # The original codes were modified. Current codes only support 'LSTM'.
        assert rnn_type == 'LSTM', 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(token_emsize if l == 0 else nhid, nhid if l != nlayers - 1 else (token_emsize if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)

        self.token_encoder = nn.Embedding(ntoken, token_emsize)
        self.decoder = nn.Linear(nhid, ntoken)
        with open(pretrained_model, 'rb') as f:
            loaded_model, _, _ = torch.load(f)
        self.load_state_dict(loaded_model.state_dict())

        if freeze_parameters:
            self.token_encoder.weight.requires_grad = False
        else:
            self.token_encoder.weight.requires_grad = True

        self.lockdrop = LockedDropout()
        self.drop = nn.Dropout(dropout)
        self.hdrop = nn.Dropout(dropouth)
        self.idrop = nn.Dropout(dropouti)


    def forward(self, token_ids, hidden, return_h=False):
        emb = embedded_dropout(self.token_encoder, token_ids, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)
        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            rnn.flatten_parameters()
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        result = output.view(output.size(0)*output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.token_emsize if self.tie_weights else self.nhid)).zero_(),
                 weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.token_emsize if self.tie_weights else self.nhid)).zero_())
                for l in range(self.nlayers)]

