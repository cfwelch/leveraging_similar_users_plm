import numpy as np
import torch


# Dropout is only applied to token_embed
def embedded_dropout(token_embed, token_ids, dropout=0.1, scale=None):
    if dropout:
      mask = token_embed.weight.data.new().resize_((token_embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(token_embed.weight) / (1 - dropout)
      masked_embed_weight = mask * token_embed.weight
      # print(mask)
    else:
      masked_embed_weight = token_embed.weight
    if scale:
      masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = token_embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    X1 = torch.nn.functional.embedding(token_ids, masked_embed_weight,
      padding_idx, token_embed.max_norm, token_embed.norm_type,
      token_embed.scale_grad_by_freq, token_embed.sparse
    )
    return X1

if __name__ == '__main__':
    vocab_size = 50
    token_emsize = 4
    bptt = 10
    batch_size = 2

    token_embed = torch.nn.Embedding(vocab_size, token_emsize)
    token_ids = np.random.random_integers(low=0, high=vocab_size-1, size=(batch_size, bptt))
    token_ids = torch.LongTensor(token_ids)

    origX = token_embed(token_ids)
    X = embedded_dropout(token_embed, token_ids)

    print(origX)
    print(X)
