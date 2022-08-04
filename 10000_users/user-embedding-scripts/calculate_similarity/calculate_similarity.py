import torch, argparse, os, pickle
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='calculate similarity score for validation users')
    parser.add_argument('--embedding_path', type=str, required=True,
                        help='input user embedding path')
    parser.add_argument('--num_training_tokens', type=int, required=True,
                        help='num training tokens')
    args = parser.parse_args()
    print(args.embedding_path)
    anchor_users = []
    with open('/local/chenxgu/present_working_directory/data_10000/anchor_users.txt', 'r') as file:
        for line in file:
            anchor_users.append(line.strip('\n'))

    validation_users = []
    with open('/local/chenxgu/present_working_directory/data_10000/validation_users.txt', 'r') as file:
        for line in file:
            validation_users.append(line.strip('\n'))

    matrix = np.zeros((len(validation_users), len(anchor_users)), dtype=np.float)
    for idx, user in enumerate(validation_users):
        v_idx = idx + len(anchor_users)
        # user_embedding is 20000 x 50
        # the first 10000 entries are anchor user embedding
        with open(os.path.join(args.embedding_path, 'model_{' + user + '}.pt'), 'rb') as f:
            model, _, _ = torch.load(f)
        user_embedding = model.user_encoder.weight.cpu().data.numpy()
        valid_user_embed = user_embedding[v_idx]
        for a_idx in range(len(anchor_users)):
            anchor_embed = user_embedding[a_idx]
            matrix[v_idx-len(anchor_users)][a_idx] = np.dot(anchor_embed, valid_user_embed) / np.linalg.norm(anchor_embed) / np.linalg.norm(valid_user_embed)

    with open('simi_matrix_' + str(args.num_training_tokens) + '.pkl', 'wb') as file:
        pickle.dump(matrix, file)


if __name__ == '__main__':
    main()

