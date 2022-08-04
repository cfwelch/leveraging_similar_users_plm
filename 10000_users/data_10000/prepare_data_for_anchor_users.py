import random, os, shutil

NUM_TRAINING_TOKEN = 20000
NUM_VALIDATION_TOKEN = 20000
NUM_TEST_TOKEN = 20000
NUM_TOKEN = NUM_TRAINING_TOKEN + NUM_VALIDATION_TOKEN + NUM_TEST_TOKEN

anchor_users = []
with open('anchor_users.txt', 'r') as file:
    for line in file:
        anchor_users.append(line.strip('\n'))

if os.path.exists(str(NUM_TOKEN) + '_tokens_of_anchor_users'):
    shutil.rmtree(str(NUM_TOKEN) + '_tokens_of_anchor_users')
os.makedirs(str(NUM_TOKEN) + '_tokens_of_anchor_users')

data_type = ['training', 'validation', 'test']
lengths = [NUM_TRAINING_TOKEN, NUM_VALIDATION_TOKEN, NUM_TEST_TOKEN ]

random.seed(20200522)
for user in anchor_users:
    print(user)
    posts = []
    with open('posts_txt/' + user, 'r') as file:
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

