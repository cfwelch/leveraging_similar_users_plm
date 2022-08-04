import random

num_tokens = 63000
num_anchor_users = 10000
valid_users = []

with open('tokens_count.txt', 'r') as file:
    for line in file:
        line = line.strip('\n').strip(' ')
        info = line.split('/')
        if len(info) > 1:
            user = info[1]
            tokens = int(info[0].split(' ')[0])
            if tokens > num_tokens:
                valid_users.append(user)

print(len(valid_users))
random.seed(1)
random.shuffle(valid_users)
"""
with open('anchor_users.txt', 'w') as file:
    for user in valid_users[:num_anchor_users]:
        file.write(user + '\n')
"""
num_validation_users = 100
with open('validation_users.txt', 'w') as file:
    for user in valid_users[num_anchor_users:num_anchor_users+num_validation_users]:
        file.write(user + '\n')
