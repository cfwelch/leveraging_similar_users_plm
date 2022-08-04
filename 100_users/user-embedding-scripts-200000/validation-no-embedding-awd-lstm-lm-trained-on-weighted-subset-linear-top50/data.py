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

        validation_users = []
        with open('../../data/validation_users.txt', 'r') as file:
            for line in file:
                validation_users.append(line.strip('\n'))

        user_idx = -1
        for idx, user in enumerate(validation_users):
            if user == self.user_name:
                user_idx = idx
                break
        assert(user_idx != -1)
        user_idx += len(self.anchor_users)

        with open('../validation-user-embedding-awd-lstm-lm-2000/model_{' + self.user_name + '}_{2000}.pt', 'rb') as f:
            temp_model, _, _ = torch.load(f)
        user_embedding = temp_model.user_encoder.weight.cpu().data.numpy()

        # user_embedding is 200 x 50
        # the first 100 entries are anchor user embedding

        val_user_embed = user_embedding[user_idx]

        self.simis = np.zeros((len(self.anchor_users),))
        for idx in range(len(self.anchor_users)):
           anchor_embed = user_embedding[idx]
           self.simis[idx] = np.dot(anchor_embed, val_user_embed) / np.linalg.norm(anchor_embed) / np.linalg.norm(val_user_embed)
        # normalize similarity
        self.simis = self.simis - np.min(self.simis)
        self.simis = self.simis / np.max(self.simis)
        print('similarity')
        print(self.simis)
        
        self.num_tokens_from_anchors = (self.simis / np.sum(self.simis) * num_training_tokens).astype(np.int)
        print('num tokens from anchors')
        print(self.num_tokens_from_anchors)
        
        self.training_token_ids = self.prepare_data('training')
        self.validation_token_ids = self.prepare_data('validation')
        self.test_token_ids = self.prepare_data('test')

    def prepare_data(self, data_type):
        top_k = 50
        print(data_type)
        if data_type == 'training':
            # prepare data from anchor users
            counter = 0
            all_posts = []
            ordered_anchor_index = np.argsort(self.num_tokens_from_anchors)[::-1]

            topsum = np.sum(np.sort(self.num_tokens_from_anchors)[::-1][:top_k])
            self.num_tokens_from_anchors = (self.num_tokens_from_anchors * self.num_training_tokens / topsum + 1).astype(np.int)
            print(ordered_anchor_index)
            current_index = 0
            while True:
                current_anchor_user = self.anchor_users[ordered_anchor_index[current_index]]
                if 200000 < self.num_tokens_from_anchors[ordered_anchor_index[current_index]]:
                    diff = self.num_tokens_from_anchors[ordered_anchor_index[current_index]] - 200000
                    remain_users_sum = np.sum(np.sort(self.num_tokens_from_anchors)[::-1][current_index+1:top_k])
                    if remain_users_sum > 0:
                        self.num_tokens_from_anchors += (self.num_tokens_from_anchors * diff / remain_users_sum + 1).astype(np.int)
                temp_num_tokens_from_anchor = min(200000, self.num_tokens_from_anchors[ordered_anchor_index[current_index]])
                temp_num_tokens_from_anchor = min(temp_num_tokens_from_anchor, self.num_training_tokens - counter)
                if temp_num_tokens_from_anchor == 0:
                    break
                
                local_counter = 0
                with open(os.path.join('../../data/250000_tokens_of_anchor_users', current_anchor_user + '_' + data_type), 'r') as file:
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
    corpus = Corpus('../GloVe-1.2-emsize-200/GloVe_200', 'wjbc', '../../data/250000_tokens_of_validation_users', num_training_tokens=2000000)
