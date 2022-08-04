import pickle
from collections import defaultdict

validation_users = []
with open('../data/validation_users.txt', 'r') as file:
    for line in file:
        validation_users.append(line.strip('\n'))

best_model_is_better = defaultdict(int)
ft_is_better = defaultdict(int)

for user in validation_users:
    with open('best_model/' + user, 'r') as file:
        best_model_data = file.readlines()

    with open('ft/' + user, 'r') as file:
        ft_data = file.readlines()

    # the last two lines are invalid
    actual_length = min(len(best_model_data), len(ft_data)) - 2

    for i in range(actual_length):
        best_model_info = best_model_data[i].strip('\n').split('\t')[0]
        best_model_info = best_model_info.split(',')
        best_model_id = int(best_model_info[0])
        try:
            best_model_rank = int(best_model_info[1])
        except:
            # rank == '?'
            # use a large number
            best_model_rank = 60000
        # best_model_nll = float(best_model_info[2])
        
        ft_info = ft_data[i].strip('\n').split('\t')[0]
        ft_info = ft_info.split(',')
        # ft_id = int(ft_info[0])
        try:
            ft_rank = int(ft_info[1])
        except:
            # rank == '?'
            # use a large number
            ft_rank = 60000
        # ft_nll = float(ft_info[2])
        # assert(best_model_id == ft_id)
        if best_model_rank == 0 and ft_rank != 0:
            best_model_is_better[best_model_id] += 1
        elif best_model_rank != 0 and ft_rank == 0:
            ft_is_better[best_model_id] += 1

sorted_best_model_is_better = sorted(best_model_is_better.items(), key=lambda kv: kv[1], reverse=True)
sorted_ft_is_better = sorted(ft_is_better.items(), key=lambda kv: kv[1], reverse=True)

with open('../user-embedding-scripts-200000/GloVe-1.2-emsize-200/GloVe_200_dictionary.pkl', 'rb') as file:
    data = pickle.load(file)

dictionary = {}
for token in data.keys():
    dictionary[data[token]] = token

sum = 0
with open('best_model_is_better.txt', 'w') as file:
    for token_id, freq in sorted_best_model_is_better:
        sum += freq
        file.write(dictionary[token_id] + '\t' + str(freq) + '\n')
    file.write('total\t' + str(sum) + '\n')

sum = 0
with open('ft_is_better.txt', 'w') as file:
    for token_id, freq in sorted_ft_is_better:
        sum += freq
        file.write(dictionary[token_id] + '\t' + str(freq) + '\n')
    file.write('total\t' + str(sum) + '\n')
