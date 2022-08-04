users=(Thorse wjbc frostyuno RobotBuddha efrique orthag mayonesa ripster55 Thief39 IConrad christ0ph TheCannon ThereisnoTruth Pinalope4Real TWFM sirbruce Zeppelanoid Zifnab25 kickstand Marvelvsdc00 DarthContinent Radico87 Aerys1 poesie GenJonesMom laddergoat89 CitizenPremier Bipolarruledout zoidberg1339 s73v3r 1338h4x TheHerbalGerbil elbruce Phoequinox ReisaD Jigsus Morganelefae EmeraldLight Lampmonster1 alekzander01 NJBilbo pics-or-didnt-happen WarPhalange nermid ameoba Nerdlinger andytuba Mace55555 mileylols ForgettableUsername)
#num_token=(100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000)
num_token=(1000)
for user in ${users[@]}
do
for n in ${num_token[@]}
do
echo ${user}
#grep "test loss" model_${user}.log
grep "test" -B 3 model_${user}.log
done
done
