users=(StabbyPants wjbc frostyuno RobotBuddha efrique orthag mayonesa ripster55 Thief39 IConrad christ0ph TheCannon ThereisnoTruth Pinalope4Real TWFM sirbruce Zeppelanoid Zifnab25 kickstand Marvelvsdc00 DarthContinent Radico87 Aerys1 poesie GenJonesMom laddergoat89 CitizenPremier Bipolarruledout zoidberg1339 s73v3r Bloodysneeze TheHerbalGerbil elbruce Phoequinox StrangerThanReality Jigsus Morganelefae EmeraldLight Lampmonster1 alekzander01 NJBilbo pics-or-didnt-happen WarPhalange MileHighBarfly ameoba Nerdlinger rainman_104 Mace55555 mileylols ForgettableUsername)
#num_token=(100 200 300 400 500 600 700 800 900 1000 2000)
num_token=(2000)
for user in ${users[@]}
do
for n in ${num_token[@]}
do
echo "python main.py --lr 3 --multiply 100 --user_name ${user} --pretrained_model ../anchor-no-embedding-awd-lstm-lm/model_2.pt --num_training_tokens_from_validation  $n --num_training_tokens_from_anchor 2000000 --data ../../data/250000_tokens_of_validation_users --pretrained_token_embedding '../GloVe-1.2-emsize-200/GloVe_200' --freeze_parameters --token_emsize 200 --batch_size 20 --dropouti 0.2 --dropouth 0.2 --dropout 0.2 --dropoute 0.1 --seed 20190930 --epoch 500 --save model.pt"
python main.py --lr 3 --multiply 100 --user_name ${user} --pretrained_model ../anchor-no-embedding-awd-lstm-lm/model_2.pt --num_training_tokens_from_validation  $n --num_training_tokens_from_anchor 2000000 --data ../../data/250000_tokens_of_validation_users --pretrained_token_embedding '../GloVe-1.2-emsize-200/GloVe_200' --freeze_parameters --token_emsize 200 --batch_size 20 --dropouti 0.2 --dropouth 0.2 --dropout 0.2 --dropoute 0.1 --seed 20190930 --epoch 500 --save model.pt > model_${user}.log
done
done
