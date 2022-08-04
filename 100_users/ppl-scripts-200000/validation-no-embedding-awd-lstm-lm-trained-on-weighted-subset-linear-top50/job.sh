#declare -a users=(StabbyPants wjbc frostyuno RobotBuddha efrique orthag mayonesa ripster55 Thief39 IConrad christ0ph TheCannon ThereisnoTruth Pinalope4Real TWFM sirbruce Zeppelanoid Zifnab25 kickstand Marvelvsdc00 DarthContinent Radico87 Aerys1 poesie GenJonesMom laddergoat89 CitizenPremier Bipolarruledout zoidberg1339 s73v3r Bloodysneeze TheHerbalGerbil elbruce Phoequinox StrangerThanReality Jigsus Morganelefae EmeraldLight Lampmonster1 alekzander01 NJBilbo pics-or-didnt-happen WarPhalange MileHighBarfly ameoba Nerdlinger rainman_104 Mace55555 mileylols ForgettableUsername)
#declare -a users=(ameoba Nerdlinger rainman_104 Mace55555 mileylols ForgettableUsername)
declare -a users=(StabbyPants wjbc frostyuno RobotBuddha)
for u in ${users[@]}
do
   user=$u
   echo "Start job with name $u"
   sbatch --job-name=$user --export=user=$user slurm_50
done

