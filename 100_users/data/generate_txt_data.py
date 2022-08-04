import os, shutil, json, collections, IPython, re

user_type = 'validation'

def replfunc(matchobj):
    # print(matchobj.group(0))
    return matchobj.group(0)[2:-8]

def count_alphabets(str):
    count = 0
    for c in str:
        if c.isalpha():
            count += 1
    return count


def main():

    if os.path.exists(user_type + '_posts_txt'):
        shutil.rmtree(user_type + '_posts_txt')
    os.makedirs(user_type + '_posts_txt')

    n_grams = [12]

    users = []
    with open(user_type + '_users.txt', 'r') as file:
        for line in file:
            users.append(line.strip('\n'))
    for user in users:
        print(user)
        valid_posts = []
        invalid_posts = []
        n_grams_dict = {}
        for n in n_grams:
            n_grams_dict[n] = collections.defaultdict(int)
        with open('posts_json/' + user + '_json_filtered_tokenized', 'r') as file:
            for line in file:
                tline = json.loads(line)
                post = tline['body'].strip('\n').strip(' ')
                post = post.encode('ascii', errors='ignore').decode()
                tokens = post.split(' ')
                # posts with small average tokens length seem usually bad
                if len(tokens) > 20 and len(post) * 1.0 / len(tokens) < 3:
                    invalid_posts.append(post + '[REASON1]')
                    continue
                # posts with super long tokens seem usually bad
                lengths = [len(token) for token in tokens]
                if max(lengths) > 30:
                    invalid_posts.append(post + '[REASON2]')
                    continue
                # clean repetition of 'url' within short sentences
                if len(tokens) < 8 and len(re.findall(r'url', post)) > 3:
                    invalid_posts.append(post + '[REASON3]')
                    continue
                # remove tables and maths
                if len(re.findall(r"\|", post)) > 3 or len(re.findall(r"\+", post)) > 3 or len(re.findall(r"\=", post)) > 3:
                    invalid_posts.append(post + '[REASON4]')
                    continue
                # remove coding related posts
                if len(re.findall(r"\([\s]*\)", post)) > 0 or len(re.findall(r"{", post)) > 0 or len(re.findall(r"}", post)) > 0:
                    invalid_posts.append(post + '[REASON5]')
                    continue
                # short posts ending with '*' usually correct a mispelling
                if len(tokens) < 5 and post.endswith('*'):
                    invalid_posts.append(post + '[REASON6]')
                    continue
                # remove posts which have short repetition
                uniques = [len(set(tokens[idx: idx+8])) for idx in range(len(tokens) - 8)]
                if uniques and min(uniques) < 4:
                    invalid_posts.append(post + '[REASON7]')
                    continue

                res = re.findall(r'\[[^\]]+\]\s\(\s/\s/\s#', post)
                if res:
                    invalid_posts.append(post + '[REASON8]')
                    continue

                # valid posts!!!
                for n in n_grams:
                    for idx in range(len(tokens) - n):
                        n_grams_dict[n][' '.join(tokens[idx: idx+n])] += 1
                valid_posts.append(tokens)

            final_posts = []
            for tokens in valid_posts:
                post = ' '.join(tokens)
                # clean repetition between sentences
                # or long repetition within sentences
                repeated = False
                for n in n_grams:
                    for idx in range(len(tokens) - n):
                        if n_grams_dict[n][' '.join(tokens[idx: idx+n])] > 3:
                            repeated = True
                            invalid_posts.append(post + '[REASON9]')
                            break
                    if repeated:
                        break
                if repeated:
                    continue

                processed_post = post.replace('subreddit _ name', 'subreddit_name')
                processed_post = processed_post.replace('reddit _ username', 'reddit_username')
                # remove emoji
                processed_post = re.sub(r'\[ \] \( / [^\)]+ \)', '', processed_post)

                # many false positive
                # processed_post = re.sub(r":\s[a-z]+\s:", '', processed_post)

                # replace hyper link of this pattern: [ sth ] ( url
                processed_post = re.sub(r'\[[^\]]+\]\s\(\surl', replfunc, processed_post)

                processed_post = processed_post.strip(' ')
                # replace multiple adjacent whitespaces with one whitespace
                processed_post = re.sub(r'[\s]+', ' ', processed_post)

                if processed_post == '':
                    invalid_posts.append(post + '[REASON10]')
                    continue

                if len(processed_post) > 50 and count_alphabets(processed_post) < len(processed_post) * 0.4:
                    invalid_posts.append(post + '[REASON11]')
                    continue

                final_posts.append(processed_post)


        with open(user_type + '_posts_txt/' + user, 'w') as file:
            for post in final_posts:
                file.write(post + ' <eos>\n')
                # file.write(post + '\n')

        with open(user_type + '_posts_txt/' + user + '_invalid', 'w') as file:
            for post in invalid_posts:
                file.write(post + '\n')

if __name__ == '__main__':
    main()


