import os, pickle, torch, random
import numpy as np

class Corpus(object):
    def __init__(self, pretrained_embedding, user_name, path, num_training_tokens_from_validation, num_validation_tokens_from_validation, num_training_tokens_from_anchor, num_similar_users):
        with open(pretrained_embedding + '_dictionary.pkl', 'rb') as file:
            self.dictionary = pickle.load(file)
        self.vocab_size = len(self.dictionary)

        self.user_name = user_name
        # note: num_training_tokens_from_anchor is the total number of training tokens from all anchor users
        self.num_training_tokens_from_anchor = num_training_tokens_from_anchor
        self.num_tokens_from_validation = {"training": num_training_tokens_from_validation,
                                           "validation": num_validation_tokens_from_validation,
                                           "test": 20000}
        self.path = path
        self.num_similar_users = num_similar_users

        self.anchor_users = []
        with open('../../data_10000/anchor_users.txt', 'r') as file:
            for line in file:
                self.anchor_users.append(line.strip('\n'))

        validation_users = []
        with open('../../data_10000/validation_users.txt', 'r') as file:
            for line in file:
                validation_users.append(line.strip('\n'))

        user_idx = -1
        for idx, user in enumerate(validation_users):
            if user == self.user_name:
                user_idx = idx
                break
        assert(user_idx != -1)

        with open("../authorship-classification/validation_similarity_matrix.pkl", 'rb') as file:
            simi_matrix = pickle.load(file)
        self.simis = simi_matrix[user_idx]

        # normalize similarity
        # self.simis = self.simis - np.min(self.simis)
        # self.simis = self.simis / np.max(self.simis)

        self.num_tokens_from_anchors = self.simis / np.sum(self.simis) * num_training_tokens_from_anchor

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
            ordered_anchor_index = np.argsort(self.num_tokens_from_anchors)[::-1]

            topsum = np.sum(np.sort(self.num_tokens_from_anchors)[::-1][:self.num_similar_users])
            self.num_tokens_from_anchors = (self.num_tokens_from_anchors * self.num_training_tokens_from_anchor / topsum + 1).astype(np.int)
            current_index = 0
            while True:
                current_anchor_user = self.anchor_users[ordered_anchor_index[current_index]]
                if 2000 < self.num_tokens_from_anchors[ordered_anchor_index[current_index]]:
                    diff = self.num_tokens_from_anchors[ordered_anchor_index[current_index]] - 2000
                    remain_users_sum = np.sum(np.sort(self.num_tokens_from_anchors)[::-1][current_index+1:self.num_similar_users])
                    if remain_users_sum > 0:
                        self.num_tokens_from_anchors += (self.num_tokens_from_anchors * diff / remain_users_sum + 1).astype(np.int)
                temp_num_tokens_from_anchor = min(2000, self.num_tokens_from_anchors[ordered_anchor_index[current_index]])
                temp_num_tokens_from_anchor = min(temp_num_tokens_from_anchor, self.num_training_tokens_from_anchor - counter)
                if temp_num_tokens_from_anchor == 0:
                    break

                local_counter = 0
                with open(os.path.join('../../data_10000/60000_tokens_of_anchor_users', current_anchor_user + '_training'), 'r') as file:
                    for post in file:
                        tokens = post.strip('\n').split(' ')
                        l = len(tokens)
                        if local_counter + l < temp_num_tokens_from_anchor:
                            all_posts.append(post)
                            local_counter += l
                        else:
                            remaining = temp_num_tokens_from_anchor - local_counter
                            if remaining > 1:
                                all_posts.append(' '.join(tokens[:remaining-1]) + ' <eos>\n')
                                local_counter += remaining
                            else:
                                all_posts.append('<eos>\n')
                                local_counter += 1
                            break
                counter += local_counter
                print(str(local_counter) + ' tokens from ' + current_anchor_user)
                current_index += 1

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
    corpus = Corpus('../../user-embedding-scripts/GloVe-1.2-emsize-200/GloVe_200', 'kcstrike', '../../data_10000/60000_tokens_of_validation_users', num_training_tokens_from_validation=2000, num_validation_tokens_from_validation=2000, num_training_tokens_from_anchor=20000, num_similar_users=10)



