
This repository contains code for our ACL 2022 paper. If you use this code, please cite:

```
@inproceedings{welch-etal-2022-leveraging,
    title = "Leveraging Similar Users for Personalized Language Modeling with Limited Data",
    author = "Welch, Charles  and
      Gu, Chenxi  and
      Kummerfeld, Jonathan K.  and
      Perez-Rosas, Veronica  and
      Mihalcea, Rada",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.122",
    doi = "10.18653/v1/2022.acl-long.122",
}
```


10000_users: This directory contains everything about the dataset of 10000 anchor users and the experiments on this dataset.

10000_users/data_10000: This directory contains everything about the dataset of 10000 anchor users.

posts_json: This directory contains tokenized posts from 15000 users.

generate_txt_data.py: This file preprocesses tokenized posts in posts_json by removing invalid posts or invalid components in posts.

posts_txt: This directory contains preprocessed valid posts from 15000 users.

posts_txt_invalid: This directory contains invalid posts from 15000 users.

tokens_count.txt: This file counts the number of valid tokens from each user.

pick_user.py: This file picks 10000 anchor users and 100 validation users from 15000 users.

anchor_users.txt: This file contains 10000 anchor users.

validation_users.txt: This file contains 100 validation users.

prepare_data_for_anchor_users.py: This file prepares training, validation and test data for anchor users.

prepare_data_for_validation_users.py: This file prepares training, validation and test data for validation users.

60000_tokens_of_anchor_users: This directory contains training, validation and test data for anchor users.

60000_tokens_of_validation_users: This directory contains training, validation and test data for validation users.

posts_of_anchor_users.txt: This file contains posts from all anchor users. It is used to train GloVe embeddings.

generate_csv.py: This file generates training/validation/test data for the authorship attribution model.

anchor_posts_and_labels_test.csv: This file contains the training data for the authorship attribution model.

anchor_posts_and_labels_training.csv: This file contains the validation data for the authorship attribution model.

anchor_posts_and_labels_validation.csv: This file contains the test data for the authorship attribution model.

validation_posts_and_labels_training.csv: This file is used to generate the similarity matrix between validation users and anchor users using the authorship attribution model.



10000_users/user-embedding-scripts: This directory contains everything about the experiments using the user embedding method.

GloVe-1.2-emsize-200: This directory contains the GloVe embeddings trained on posts from anchor users.

anchor-no-embedding-awd-lstm-lm: Files in this directory train a standard language model on 10000 anchor users. The trained model is model_2.pt. 

anchor-user-embedding-awd-lstm-lm: Files in this directory train a language model with user embedding on 10000 anchor users. The trained model is model_2.pt.

validation-user-embedding-awd-lstm-lm-2000: Files in this directory train user embedding for validation users using 2000 tokens from each validation user.

calculate_similarity: Files in this directory calculate the similarity matrix between validation users and anchor users using the cosine similarity between the user embeddings (from anchor-user-embedding-awd-lstm-lm and validation-user-embedding-awd-lstm-lm-2000). The similarity matrix is in simi_matrix_2000.pkl. It is not normalized. The numbers in this matrix are from -1 to 1.

validation-no-embedding-awd-lstm-lm-finetune-on-validation-only-2000: Files in this directory fine-tune the pre-trained model (from anchor-no-embedding-awd-lstm-lm) on 2000 tokens from each validation user. The results are used as our baseline.

The following directories contain similar experiments with different hyper parameters. Take validation-no-embedding-awd-lstm-lm-finetune-on-validation-and-anchor-two-step-100000 as an example. For each validation user: it takes the pre-trained model from anchor-no-embedding-awd-lstm-lm, it first fine-tunes the model on 100000 tokens from the validation user’s similar anchor users, then fine-tunes the model on 2000 tokens from this validation user. The similar anchor users of a validation user is determined by simi_matrix_2000.pkl from calculate_similarity. There are two important hyper parameters in preparing the training data for the first fine-tuning step: num_training_tokens_from_anchor is used to determine the number of tokens from similar anchor users (which is 100000 in this case), num_similar_users is used to determine the number of anchor users those tokens come from, it should be at least num_training_tokens_from_anchor / 2000 (which is 50 in this case) because we only have 2000 training tokens from each anchor user. These two hyper parameters can be modified in run.sh

validation-no-embedding-awd-lstm-lm-finetune-on-validation-and-anchor-two-step-20000

validation-no-embedding-awd-lstm-lm-finetune-on-validation-and-anchor-two-step-40000

validation-no-embedding-awd-lstm-lm-finetune-on-validation-and-anchor-two-step-60000

validation-no-embedding-awd-lstm-lm-finetune-on-validation-and-anchor-two-step-80000

validation-no-embedding-awd-lstm-lm-finetune-on-validation-and-anchor-two-step-100000

validation-no-embedding-awd-lstm-lm-finetune-on-validation-and-anchor-two-step-180000

validation-no-embedding-awd-lstm-lm-finetune-on-validation-and-anchor-two-step-200000

validation-no-embedding-awd-lstm-lm-finetune-on-validation-and-anchor-two-step-300000

validation-no-embedding-awd-lstm-lm-finetune-on-validation-and-anchor-two-step-400000



10000_users/authorship-attribution-scripts: This directory contains everything about the experiments using the authorship attribution method.

authorship-classification: This files contains everything about the authorship attribution model.

main.py: This file trains an authorship attribution model on anchor users (2000 training tokens, 2000 validation tokens and 200000 test tokens from each anchor user).

run.sh: This file is used to run main.py with arguments. Nine combinations of hyper parameters were tried. model_9.pt is the best. Its hyper parameters are recorded in output_9.txt

glove.6B.200d.txt.pt and glove.6B.200d.txt are used for main.py.

generate_similarity_matrix_anchor.py: This file generates the similarity matrix between anchor users using the test data.

generate_similarity_matrix_validation.py: This file generates the similarity matrix between validation users and anchor users using 2000 tokens from each validation user.

anchor_similarity_matrix.pkl: This file contains the similarity matrix between anchor users generated by generate_similarity_matrix_anchor.py using model_9.pt.

validation_similarity_matrix.pkl: This file contains the similarity matrix between validation users and anchor users generated by generate_similarity_matrix_validation.py using model_9.pt.

