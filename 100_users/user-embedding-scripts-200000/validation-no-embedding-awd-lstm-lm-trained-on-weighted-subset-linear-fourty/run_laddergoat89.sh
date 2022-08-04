#users=(StabbyPants wjbc frostyuno RobotBuddha efrique orthag mayonesa ripster55 Thief39 IConrad christ0ph TheCannon ThereisnoTruth Pinalope4Real TWFM sirbruce Zeppelanoid Zifnab25 kickstand Marvelvsdc00 DarthContinent Radico87 Aerys1 poesie GenJonesMom laddergoat89 CitizenPremier Bipolarruledout zoidberg1339 s73v3r Bloodysneeze TheHerbalGerbil elbruce Phoequinox StrangerThanReality Jigsus Morganelefae EmeraldLight Lampmonster1 alekzander01 NJBilbo pics-or-didnt-happen WarPhalange MileHighBarfly ameoba Nerdlinger rainman_104 Mace55555 mileylols ForgettableUsername)
users=(laddergoat89)
for user in ${users[@]}
do
echo "python main.py --lr 3 --user_name ${user} --num_training_tokens 8000000 --data ../../data/250000_tokens_of_validation_users --pretrained_token_embedding '../GloVe-1.2-emsize-200/GloVe_200' --freeze_parameters --token_emsize 200 --batch_size 20 --dropouti 0.2 --dropouth 0.2 --dropout 0.2 --dropoute 0.1 --seed 20190930 --epoch 200 --save model_${user}.pt"
python main.py --lr 3 --user_name ${user} --num_training_tokens 8000000 --data ../../data/250000_tokens_of_validation_users --pretrained_token_embedding '../GloVe-1.2-emsize-200/GloVe_200' --freeze_parameters --token_emsize 200 --batch_size 20 --dropouti 0.2 --dropouth 0.2 --dropout 0.2 --dropoute 0.1 --seed 20190930 --epoch 200 --save model_${user}.pt > model_${user}.log
done
