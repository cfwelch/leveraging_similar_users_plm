import random, os, shutil

NUM_TOKEN = 250000
TRAINING_PERCENTAGE = 0.8
VALIDATION_PERCENTAGE = 0.1
TEST_PERCENTAGE = 0.1

anchor_users = []
with open('anchor_users.txt', 'r') as file:
    for line in file:
        anchor_users.append(line.strip('\n'))

if os.path.exists(str(NUM_TOKEN) + '_tokens_of_anchor_users'):
    shutil.rmtree(str(NUM_TOKEN) + '_tokens_of_anchor_users')
os.makedirs(str(NUM_TOKEN) + '_tokens_of_anchor_users')

data_type = ['training', 'validation', 'test']
lengths = [int(NUM_TOKEN * TRAINING_PERCENTAGE),
           int(NUM_TOKEN * VALIDATION_PERCENTAGE),
           int(NUM_TOKEN * TEST_PERCENTAGE)]

random.seed(20190930)
for user in anchor_users:
    print(user)
    posts = []
    with open('anchor_posts_txt/' + user, 'r') as file:
        for line in file:
            posts.append(line)
    random.shuffle(posts)
    data_type_index = 0
    counter = 0
    chosen_posts = []
    for post in posts:
        tokens = post.strip('\n').split(' ')
        l = len(tokens)

        if counter + l < lengths[data_type_index]:
            chosen_posts.append(post)
            counter += l
        else:
            remaining = lengths[data_type_index] - counter
            if remaining > 1:
                chosen_posts.append(' '.join(tokens[:remaining-1]) + ' <eos>\n')
            else:
                chosen_posts.append('<eos>\n')
            counter += remaining
            with open(str(NUM_TOKEN) + '_tokens_of_anchor_users/' + user + '_' + data_type[data_type_index], 'w') as file:
                for post in chosen_posts:
                    file.write(post)

            if data_type_index == 2:
                break
            else:
                data_type_index += 1
                counter = 0
                chosen_posts = []

