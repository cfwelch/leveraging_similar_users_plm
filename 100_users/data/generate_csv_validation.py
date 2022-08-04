import csv

num_tokens = 500

validation_users = []
with open('validation_users.txt', 'r') as file:
    for line in file:
        validation_users.append(line.strip('\n'))

with open('validation_posts_and_labels_training_' + str(num_tokens) + '.csv', mode='w') as csv_file:
    fieldnames = ['post', 'author']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter='\t')

    writer.writeheader()
    for index, user in enumerate(validation_users):
        print(user)
        count = 0
        with open('250000_tokens_of_validation_users/' + user + '_training', 'r') as file:
            for line in file:
                tokens = line.strip('\n').split(' ')
                length = len(tokens)
                if count + length < num_tokens:
                    writer.writerow({'post': line.strip('\n'), 'author': index+100})
                    count += length
                elif count + length == num_tokens:
                    writer.writerow({'post': line.strip('\n'), 'author': index+100})
                    break
                else:
                    writer.writerow({'post': ' '.join(tokens[:num_tokens-count]), 'author': index+100})
                    break

