import os, pickle, torch, random

class Corpus(object):
    def __init__(self, pretrained_embedding, users_type, path, num_training_token_per_user):
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
        self.num_tokens_per_user = {'training': num_training_token_per_user,
                                    'validation': int(num_training_token_per_user * 0.25),
                                    'test': 25000}

        self.training_token_ids = self.prepare_data('training')
        self.validation_token_ids = self.prepare_data('validation')
        self.test_token_ids = self.prepare_data('test')

    def prepare_data(self, data_type):
        token_ids = torch.LongTensor(len(self.users) * self.num_tokens_per_user[data_type])

        all_posts = []
        for user_index, user in enumerate(self.users):
            counter = 0
            with open(os.path.join(self.path, user + '_' + data_type), 'r') as file:
                for line in file:
                    tokens = line.strip('\n').split(' ')
                    l = len(tokens)
                    if counter + l < self.num_tokens_per_user[data_type]:
                        all_posts.append(line)
                        counter += l
                    else:
                        remaining = self.num_tokens_per_user[data_type] - counter
                        if remaining > 1:
                            all_posts.append(' '.join(tokens[:remaining-1]) + ' <eos>\n')
                            counter += remaining
                        else:
                            all_posts.append('<eos>\n')
                            counter += 1
                        break

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
    corpus = Corpus('../GloVe-1.2-emsize-200/GloVe_200', 'anchor_users', '../../data/250000_tokens_of_anchor_users', 20)


if __name__ == '__main__':
    main()

