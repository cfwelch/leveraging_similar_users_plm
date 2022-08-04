import os, pickle, torch

class Corpus(object):
    def __init__(self, pretrained_embedding, user_name, num_training_tokens, num_validation_tokens, path):
        with open(pretrained_embedding + '_dictionary.pkl', 'rb') as file:
            self.dictionary = pickle.load(file)
        self.vocab_size = len(self.dictionary)

        self.user_name = user_name
        self.valid_users = []
        with open('/local/chenxgu/present_working_directory/data_10000/validation_users.txt', 'r') as file:
            for line in file:
                self.valid_users.append(line.strip('\n'))
        for index, u in enumerate(self.valid_users):
            if u == user_name:
                self.user_index = index
        # user_index starts from anchor users
        self.user_index += 10000
        print(self.user_name, self.user_index)

        self.num_tokens = {'training': num_training_tokens,
                           'validation': num_validation_tokens,
                           'test': 20000}
        self.path = path

        self.training_token_ids, self.training_user_ids = self.prepare_data('training')
        self.validation_token_ids, self.validation_user_ids = self.prepare_data('validation')
        self.test_token_ids, self.test_user_ids = self.prepare_data('test')

    def prepare_data(self, data_type):
        counter = 0
        posts = []
        with open(os.path.join(self.path, self.user_name + '_' + data_type), 'r') as file:
            for line in file:
                tokens = line.strip('\n').split(' ')
                l = len(tokens)
                if counter + l < self.num_tokens[data_type]:
                    posts += tokens
                    counter += l
                else:
                    remaining = self.num_tokens[data_type] - counter
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


def main():
    corpus = Corpus('../GloVe-1.2-emsize-200/GloVe_200', 'Tipop', 500, 500, '/local/chenxgu/present_working_directory/data_10000/60000_tokens_of_validation_users')


if __name__ == '__main__':
    main()

