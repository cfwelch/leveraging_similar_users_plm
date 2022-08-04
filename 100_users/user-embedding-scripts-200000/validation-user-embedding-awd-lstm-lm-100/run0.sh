user=(StabbyPants wjbc frostyuno RobotBuddha efrique orthag mayonesa ripster55 Thief39 IConrad christ0ph TheCannon ThereisnoTruth Pinalope4Real TWFM sirbruce Zeppelanoid Zifnab25 kickstand Marvelvsdc00 DarthContinent Radico87 Aerys1 poesie GenJonesMom laddergoat89 CitizenPremier Bipolarruledout zoidberg1339 s73v3r Bloodysneeze TheHerbalGerbil elbruce Phoequinox StrangerThanReality Jigsus Morganelefae EmeraldLight Lampmonster1 alekzander01 NJBilbo pics-or-didnt-happen WarPhalange MileHighBarfly ameoba Nerdlinger rainman_104 Mace55555 mileylols ForgettableUsername)
#num_token=(100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 200000 300000 400000)
num_token=(100)
for u in ${user[@]}
do
for i in ${num_token[@]}
do
echo "python main.py --lr 3 --user_name $u --num_training_token $i --data ../../data/250000_tokens_of_validation_users --pretrained_token_embedding '../../scripts/GloVe-1.2-emsize-200/GloVe_200' --freeze_parameters --token_emsize 200 --user_emsize 50 --batch_size 20 --dropouti 0.2 --dropouth 0.2 --dropout 0.2 --dropoute 0.1 --seed 20190930 --epoch 500 --save model_{$u}_{$i}.pt"
python main.py --lr 3 --user_name $u --num_training_token $i --data ../../data/250000_tokens_of_validation_users --pretrained_token_embedding '../../scripts/GloVe-1.2-emsize-200/GloVe_200' --freeze_parameters --token_emsize 200 --user_emsize 50 --batch_size 20 --dropouti 0.2 --dropouth 0.2 --dropout 0.2 --dropoute 0.1 --seed 20190930 --epoch 500 --save model_{$u}_{$i}.pt > model_{$u}_{$i}.log
done
done
