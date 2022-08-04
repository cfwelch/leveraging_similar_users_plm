import csv

anchor_users = []
with open('anchor_users.txt', 'r') as file:
    for line in file:
        anchor_users.append(line.strip('\n'))

with open('anchor_posts_and_labels_validation.csv', mode='w') as csv_file:
    fieldnames = ['post', 'author']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter='\t')

    writer.writeheader()
    for index, user in enumerate(anchor_users):
        print(user)
        with open('250000_tokens_of_anchor_users/' + user + '_validation', 'r') as file:
            for line in file:
                writer.writerow({'post': line.strip('\n'), 'author': index})
