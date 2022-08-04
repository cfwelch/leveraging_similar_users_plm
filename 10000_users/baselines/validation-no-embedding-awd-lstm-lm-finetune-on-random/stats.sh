user=(red_firetruck Shampyon bubba9999 aMalfunction GamiSB idontevenknobro mandycane18 E3K laschupacabras benshere someguy73 Fruloops riotingchimps femaiden PhairyFeenix iateyourbees kcstrike Tyranith owenaise XFX_Samsung JohnnyValet 316trees whitecrane deviationblue Trout_Tickler IWantAFuckingUsename PurpleSharkShit begrudged qtipvesto vajav Saganic DarKnightofCydonia AerThreepwood IAMTHEDEATHMACHINE rockets_meowth Neokev WillyTanner topcat5 Titty_PMs_Please Dw-Im-Here AmunRa666 cookie75 TelstarGlitch FaKeShAdOw RedPandaJr Hikari_Oshiro is-this-kosher free_at_last drwolffe cube1234567890 Duals902 mrpeabody208 lbruiser Wyatt1313 cmVkZGl0 kairedfern rawrtherapy BurntLeftovers Dunbeezy UNSCNova Fs0i ramstoria waterbed87 Tipop Thurgood_Marshall cleverlyannoying Mohl97 trekkie80 13ren the_singular_anyone ThePKAHistorian The_A_Drain TheNoveltyAccountant elcapitansmirk venomousbeetle djowen68 havebananas MrCraftLP The_Holy_Bison yosemitesquint ATtheorytime f1manoz Caedus leafystormclouds840 KPexEA iateone rob79 S11008 Taodeist Maxtrt stups317 7Snakes PWNASAURAUSREX E10DIN nyran20 ClimateMom Hllblzr310 Rougey REDDITSHITLORD jalkloben)
for u in ${user[@]}
do
# echo $u
# grep "test ppl" model_$u.log
grep "Saving" -B 3 model_$u.log | tail -3
# grep "epoch" model_$u.log | tail -1
done
