import torch


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(token_ids, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = token_ids.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    token_ids = token_ids.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    token_ids = token_ids.view(bsz, -1).t().contiguous()
    if args.cuda:
        token_ids = token_ids.cuda()
    return token_ids


def get_batch(token_ids, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(token_ids) - 1 - i)
    input = token_ids[i:i+seq_len]
    target = token_ids[i+1:i+1+seq_len].view(-1)
    return input, target
