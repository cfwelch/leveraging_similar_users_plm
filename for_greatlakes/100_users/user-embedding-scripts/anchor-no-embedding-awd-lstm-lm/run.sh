python main.py --lr 3 --users_type anchor_users --data ../../data/250000_tokens_of_anchor_users --pretrained_token_embedding '../GloVe-1.2-emsize-200/GloVe_200' --freeze_parameters --token_emsize 200 --batch_size 20 --dropouti 0.2 --dropouth 0.2 --dropout 0.2 --dropoute 0.1 --seed 20190930 --epoch 200 --save model_2.pt > model_2.log

