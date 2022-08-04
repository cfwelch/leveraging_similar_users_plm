import os, pickle, torch, random
import numpy as np

class Corpus(object):
    def __init__(self, pretrained_embedding, user_name, path, num_training_tokens_from_validation, num_validation_tokens_from_validation, num_anchor_users):
        with open(pretrained_embedding + '_dictionary.pkl', 'rb') as file:
            self.dictionary = pickle.load(file)
        self.vocab_size = len(self.dictionary)

        self.user_name = user_name
        self.num_tokens_from_validation = {"training": num_training_tokens_from_validation,
                                           "validation": num_validation_tokens_from_validation,
                                           "test": 25000}
        self.path = path
        self.num_anchor_users = num_anchor_users

        self.anchor_users = []
        with open('../../data/anchor_users.txt', 'r') as file:
            for line in file:
                self.anchor_users.append(line.strip('\n'))

        validation_users = []
        with open('../../data/validation_users.txt', 'r') as file:
            for line in file:
                validation_users.append(line.strip('\n'))

        self.training_token_ids_anchor = self.prepare_data('training_anchor')
        self.training_token_ids = self.prepare_data('training')
        self.validation_token_ids = self.prepare_data('validation')
        self.test_token_ids = self.prepare_data('test')

    def prepare_data(self, data_type):
        print(data_type)
        if data_type == 'training_anchor':
            # prepare data from anchor users
            counter = 0
            all_posts = []

            num_tokens_from_each_anchor = 200000

            random_seed = np.sum(np.array([ord(c) for c in self.user_name]))
            random.seed(random_seed)
            chosen_anchor_users = random.choices(self.anchor_users, k=self.num_anchor_users)

            for current_anchor_user in chosen_anchor_users:
                local_counter = 0
                with open(os.path.join('../../data/250000_tokens_of_anchor_users', current_anchor_user + '_training'), 'r') as file:
                    for post in file:
                        tokens = post.strip('\n').split(' ')
                        l = len(tokens)
                        if local_counter + l < num_tokens_from_each_anchor:
                            all_posts.append(post)
                            local_counter += l
                        else:
                            remaining = num_tokens_from_each_anchor - local_counter
                            if remaining > 1:
                                all_posts.append(' '.join(tokens[:remaining-1]) + ' <eos>\n')
                                local_counter += remaining
                            else:
                                all_posts.append('<eos>\n')
                                local_counter += 1
                            break
                counter += local_counter
                print(str(local_counter) + ' tokens from ' + current_anchor_user)

            token_ids = torch.LongTensor(counter)

        else:
            counter = 0
            all_posts = []
            # prepare data from validation user
            with open(os.path.join(self.path, self.user_name + '_' + data_type), 'r') as file:
                for post in file:
                    tokens = post.strip('\n').split(' ')
                    l = len(tokens)
                    if counter + l < self.num_tokens_from_validation[data_type]:
                        all_posts.append(post)
                        counter += l
                    else:
                        remaining = self.num_tokens_from_validation[data_type] - counter
                        if remaining > 1:
                           all_posts.append(' '.join(tokens[:remaining-1]) + ' <eos>\n')
                        else:
                           all_posts.append('<eos>\n')
                        break
            token_ids = torch.LongTensor(self.num_tokens_from_validation[data_type])

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


if __name__ == "__main__":
    corpus = Corpus('../../user-embedding-scripts/GloVe-1.2-emsize-200/GloVe_200', 'poesie', '../../data/250000_tokens_of_validation_users', num_training_tokens_from_validation=2000, num_validation_tokens_from_validation=25000, num_anchor_users=5)

