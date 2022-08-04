import csv

num_tokens = 2000
user_type = "validation"
data_type = "training"
idx_base = {"anchor": 0,
            "validation": 10000}

users = []
with open(user_type + '_users.txt', 'r') as file:
    for line in file:
        users.append(line.strip('\n'))

with open(user_type + '_posts_and_labels_' + data_type + '.csv', mode='w') as csv_file:
    fieldnames = ['post', 'author']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter='\t')

    writer.writeheader()
    for index, user in enumerate(users):
        print(user)
        count = 0
        with open('60000_tokens_of_' + user_type + '_users/' + user + '_' + data_type, 'r') as file:
            for line in file:
                tokens = line.strip('\n').split(' ')
                length = len(tokens)
                if count + length < num_tokens:
                    writer.writerow({'post': line.strip('\n'), 'author': index+idx_base[user_type]})
                    count += length
                elif count + length == num_tokens:
                    writer.writerow({'post': line.strip('\n'), 'author': index+idx_base[user_type]})
                    break
                else:
                    writer.writerow({'post': ' '.join(tokens[:num_tokens-count]), 'author': index+idx_base[user_type]})
                    break

