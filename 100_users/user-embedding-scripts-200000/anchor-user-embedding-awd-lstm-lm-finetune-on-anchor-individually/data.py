import os, pickle, torch

class Corpus(object):
    def __init__(self, pretrained_embedding, user_name, num_training_tokens, path):
        with open(pretrained_embedding + '_dictionary.pkl', 'rb') as file:
            self.dictionary = pickle.load(file)
        self.vocab_size = len(self.dictionary)

        self.user_name = user_name
        self.anchor_users = []
        with open('../../data/anchor_users.txt', 'r') as file:
            for line in file:
                self.anchor_users.append(line.strip('\n'))
        for index, u in enumerate(self.anchor_users):
            if u == user_name:
                self.user_index = index
        print(self.user_name, self.user_index)

        self.num_training_tokens = num_training_tokens

        self.path = path

        self.training_token_ids, self.training_user_ids = self.prepare_data('training')
        self.validation_token_ids, self.validation_user_ids = self.prepare_data('validation')
        self.test_token_ids, self.test_user_ids = self.prepare_data('test')

    def prepare_data(self, data_type):
        if data_type == 'training':
            counter = 0
            posts = []
            with open(os.path.join(self.path, self.user_name + '_' + data_type), 'r') as file:
                for line in file:
                    tokens = line.strip('\n').split(' ')
                    l = len(tokens)
                    if counter + l < self.num_training_tokens:
                        posts += tokens
                        counter += l
                    else:
                        remaining = self.num_training_tokens - counter
                        if remaining > 1:
                            posts += tokens[:remaining-1] + ['<eos>']
                        else:
                            posts += ['<eos>']
                        break
            print(data_type)
            print(len(posts))

            token_ids = torch.LongTensor(len(posts))
            user_ids = torch.LongTensor(len(posts))
            for index, p in enumerate(posts):
                token_ids[index] = self.dictionary.get(p, self.vocab_size-1)
                user_ids[index] = self.user_index
            return token_ids, user_ids

        else:
            tokens = 0
            with open(os.path.join(self.path, self.user_name + '_' + data_type), 'r') as file:
                for line in file:
                    words = line.strip('\n').split(' ')
                    tokens += len(words)
            print(data_type)
            print(tokens)

            token_ids = torch.LongTensor(tokens)
            user_ids = torch.LongTensor(tokens)
            token = 0
            with open(os.path.join(self.path, self.user_name + '_' + data_type), 'r') as file:
                for line in file:
                    words = line.strip('\n').split(' ')
                    for word in words:
                        # the default token index is self.vocab_size - 1 ('<unk>')
                        token_ids[token] = self.dictionary.get(word, self.vocab_size-1)
                        user_ids[token] = self.user_index
                        token += 1
            return token_ids, user_ids


def main():
    corpus = Corpus('../GloVe-1.2-emsize-200/GloVe_200', 'matts2', 200000, '../../data/250000_tokens_of_anchor_users')


if __name__ == '__main__':
    main()

