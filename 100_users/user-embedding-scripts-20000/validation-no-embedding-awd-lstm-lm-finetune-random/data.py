import os, pickle, torch

class Corpus(object):
    def __init__(self, pretrained_embedding, user_name, num_training_tokens, path):
        with open(pretrained_embedding + '_dictionary.pkl', 'rb') as file:
            self.dictionary = pickle.load(file)
        self.vocab_size = len(self.dictionary)

        self.user_name = user_name
        self.num_tokens = {'training': num_training_tokens,
                           'validation': int(num_training_tokens * 0.25),
                           'test': 25000}
        self.path = path

        self.training_token_ids = self.prepare_data('training')
        self.validation_token_ids = self.prepare_data('validation')
        self.test_token_ids = self.prepare_data('test')

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
        for index, p in enumerate(posts):
            token_ids[index] = self.dictionary.get(p, self.vocab_size-1)
        return token_ids



if __name__ == '__main__':
    corpus = Corpus('../GloVe-1.2-emsize-200/GloVe_200', 'wjbc', 1000, '../../data/250000_tokens_of_validation_users')


