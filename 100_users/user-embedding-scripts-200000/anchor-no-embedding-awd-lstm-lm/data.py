import os, pickle, torch, random

class Corpus(object):
    def __init__(self, pretrained_embedding, users_type, path):
        with open(pretrained_embedding + '_dictionary.pkl', 'rb') as file:
            self.dictionary = pickle.load(file)
        self.vocab_size = len(self.dictionary)
        self.users = []
        with open('../../data/' + users_type + '.txt', 'r') as file:
            for line in file:
                self.users.append(line.strip('\n'))
        print(users_type)
        print(self.users)

        self.path = path

        self.training_token_ids = self.prepare_data('training')
        self.validation_token_ids = self.prepare_data('validation')
        self.test_token_ids = self.prepare_data('test')

    def prepare_data(self, data_type):
        tokens = 0
        for user in self.users:
            with open(os.path.join(self.path, user + '_' + data_type), 'r') as file:
                for line in file:
                    words = line.strip('\n').split(' ')
                    tokens += len(words)

        token_ids = torch.LongTensor(tokens)

        all_posts = []
        for user_index, user in enumerate(self.users):
            with open(os.path.join(self.path, user + '_' + data_type), 'r') as file:
                temp = file.readlines()
                all_posts += temp

        indices_list = list(range(len(all_posts)))
        random.seed(100)
        random.shuffle(indices_list)

        token = 0
        for i in indices_list:
            words = all_posts[i].strip('\n').split(' ')
            for word in words:
                # the default token index is self.vocab_size - 1 ('<unk>')
                token_ids[token] = self.dictionary.get(word, self.vocab_size-1)
                token += 1

        print(token)
        return token_ids


def main():
    corpus = Corpus('../GloVe-1.2-emsize-200/GloVe_200', 'anchor_users', '../../data/500000_tokens_of_anchor_users')


if __name__ == '__main__':
    main()

