import os, pickle, torch, random
import numpy as np

class Corpus(object):
    def __init__(self, pretrained_embedding, user_name, path, num_training_tokens=2000000):
        with open(pretrained_embedding + '_dictionary.pkl', 'rb') as file:
            self.dictionary = pickle.load(file)
        self.vocab_size = len(self.dictionary)

        self.user_name = user_name
        # note: num_training_tokens is the total number of training tokens from all anchor users
        self.num_training_tokens = num_training_tokens
        self.path = path

        self.anchor_users = []
        with open('../../data/anchor_users.txt', 'r') as file:
            for line in file:
                self.anchor_users.append(line.strip('\n'))

        self.training_token_ids = self.prepare_data('training')
        self.validation_token_ids = self.prepare_data('validation')
        self.test_token_ids = self.prepare_data('test')

    def prepare_data(self, data_type):
        print(data_type)
        random_seed = np.sum(np.array([ord(c) for c in self.user_name]))
        random_seed = random_seed % 1000
        random.seed(random_seed)
        if data_type == 'training':
            print('random seed', random_seed)
            all_posts = []
            for user_index, user in enumerate(self.anchor_users):
                counter = 0
                with open(os.path.join('../../data/250000_tokens_of_anchor_users', user + '_' + data_type), 'r') as file:
                    for post_idx, post in enumerate(file):
                        if post_idx < random_seed:
                            continue

                        tokens = post.strip('\n').split(' ')
                        l = len(tokens)
                        if counter + l < self.num_training_tokens // len(self.anchor_users):
                            all_posts.append(post)
                            counter += l
                        else:
                            remaining = self.num_training_tokens // len(self.anchor_users) - counter
                            if remaining > 1:
                                all_posts.append(' '.join(tokens[:remaining-1]) + ' <eos>\n')
                                counter += remaining
                            else:
                                all_posts.append('<eos>\n')
                                counter += 1
                            break
                    assert(counter == self.num_training_tokens // len(self.anchor_users))
            token_ids = torch.LongTensor(self.num_training_tokens)

        # validation data or test data
        else:
            tokens = 0
            with open(os.path.join(self.path, self.user_name + '_' + data_type), 'r') as file:
                for line in file:
                    words = line.strip('\n').split(' ')
                    tokens += len(words)
            token_ids = torch.LongTensor(tokens)

            all_posts = []
            with open(os.path.join(self.path, self.user_name + '_' + data_type), 'r') as file:
                all_posts = file.readlines()

        indices_list = list(range(len(all_posts)))
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


if __name__ == "__main__":
    corpus = Corpus('../GloVe-1.2-emsize-200/GloVe_200', 'wjbc', '../../data/250000_tokens_of_validation_users', num_training_tokens=2000000)


