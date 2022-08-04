import torch, os, pickle
import numpy as np

def main():
    # user_embedding is 200 x 50
    # the first 100 entries are anchor user embedding
    with open('../anchor-user-embedding-awd-lstm-lm/model_1.pt', 'rb') as f:
        model, _, _ = torch.load(f)
    user_embedding = model.user_encoder.weight.cpu().data.numpy()  
    matrix = np.zeros((100,100), dtype=np.float)
    for u_idx_1 in range(100):
        user_embed_1 = user_embedding[u_idx_1]
        for u_idx_2 in range(100):
            user_embed_2 = user_embedding[u_idx_2]
            matrix[u_idx_1][u_idx_2] = np.dot(user_embed_1, user_embed_2) / np.linalg.norm(user_embed_1) / np.linalg.norm(user_embed_2)

    with open('anchor_simi_matrix.pkl', 'wb') as file:
        pickle.dump(matrix, file)


if __name__ == '__main__':
    main()
