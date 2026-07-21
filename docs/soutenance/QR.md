# Dossier de préparation Q&R - Soutenance M1 Autonomous Twin

**Simulation d'un jury sceptique** (un prof IA/Big Data, un prof architecture logicielle).
Objectif : anticiper les 15 minutes de questions. Chaque question naît de ce que le
jury AURA VU sur les slides ou de ce qui y manque. Les réponses sont ancrées dans les
faits exacts du repo (fichiers, commits, chiffres) et privilégient l'honnêteté qui
transforme une faiblesse en preuve de maturité.

**Porteurs.** Charles = simulateur / physique / archi. Nohlan = U-Net / CNN conduite /
datasets conduite. Lorenzo = panneaux / optimisation / méthode.

**Renvoi.** Annexe du deck qui appuie la réponse (A1 pas-de-RL, A2 classe aucun/murs,
A3 variance/repro, A4 barème, A5 protocole de mesure, A6 méthode/décisions), ou « aucune ».

**Règle transverse.** Le deck est construit avec plusieurs appâts d'honnêteté (slide 5
IoU, slide 9 occlusion, slide 13 plafond déplacé). Un jury sceptique mord dessus : ces
appâts sont des forces SI on les assume, des pièges si on se défend.

---

## A. Architecture et simulation

### A1. Qu'est-ce qui fait un « jumeau numérique » et pas un jeu vidéo 2D ?
> « Vous appelez ça un jumeau numérique, mais techniquement c'est un simulateur 2D vu de
> dessus avec une physique bicycle simplifiée. Qu'est-ce qui en fait un jumeau numérique ? »

**Réponse.** Le terme désigne ici une propriété précise et vérifiable, pas une ambition
de photoréalisme : deux processus strictement séparés (ZeroMQ REQ/REP) où le pilote n'a
accès à AUCUN état interne du simulateur, uniquement ce qui transite sur le réseau
(caméra 128 px, lidar 5 rayons, vitesse). C'est ce cloisonnement qui rend l'exercice
honnête : l'IA ne peut pas tricher en lisant la vérité terrain, comme un vrai véhicule
ne connaît pas la carte. On assume que la physique est une bicycle 2D (les 4 équations
du sujet) et que l'échelle km/h repose sur une constante encore marquée « à calibrer »
(`SPEED_SCALE=0.72`, physics.py:38) : le jumeau est celui de la BOUCLE
perception-décision-action, pas du châssis.
**Porteur.** Charles.
**Renvoi.** aucune.

### A2. Le pilote est bloquant sur ZMQ : est-ce vraiment deux mondes indépendants ?
> « Votre schéma montre deux process indépendants, mais le simulateur bloque en attendant
> la réponse du pilote. Si votre CNN met 100 ms, tout ralentit. C'est découplé, vraiment ? »

**Réponse.** C'est synchrone par choix : REQ/REP garantit une réponse-commande par
requête-capteurs, donc zéro dérive d'horloge entre les deux mondes, mais oui, le coût
d'inférence du pilote s'ajoute au pas de temps du simulateur. On l'a payé cash le 24/04 :
brancher le CNN a fait exploser le timeout serveur de 100 ms (calibré pour un PID) et il
a fallu le passer à 5 s pour absorber le premier forward-pass PyTorch à froid (commit
`ac44244`, 32 min après l'ajout du CNN `9f352a5`). Le découplage est architectural (deux
binaires, le simu reste jouable au clavier sans pilote), pas temps réel : une
extrapolation d'état découplée serait un chantier M2, sans bénéfice pour une démo à
60 Hz stables.
**Porteur.** Charles.
**Renvoi.** aucune.

### A3. Physique dépendante du framerate vs banc à dt fixe : que mesurez-vous ?
> « Votre friction s'applique par frame, pas au dt : votre physique dépend donc du
> framerate. Le banc tourne à dt fixe 1/60, la démo à FPS variable. Vos chiffres décrivent
> quoi, alors, la démo ou une simulation idéale ? »

**Réponse.** C'est le point technique le plus juste qu'on puisse nous opposer, et il est
déjà documenté chez nous. Oui : `FRICTION` (2 %) est appliquée par frame et non scalée
par dt (physics.py:60), donc la vitesse d'équilibre dépend du FPS (98 px/s à 60 fps,
49 px/s à 120). Le banc tourne justement à dt fixe 1/60 pour être DÉTERMINISTE et
reproductible : c'est un choix, pas un oubli, il mesure le potentiel de la politique de
contrôle indépendamment du CPU. La démo tourne à 54-58 FPS stables mesurés (dashboard
compris), donc dans le même régime que le banc ; la physique framerate-indépendante
(`friction^dt`) est explicitement le premier chantier M2 chiffré, on la connaît, on ne
la cache pas.
**Porteur.** Charles (physique) et Lorenzo (banc).
**Renvoi.** A5.

### A4. La caméra est un crop aligné monde, pas un capteur embarqué : raccourci ?
> « Votre caméra est un crop de l'écran aligné sur le monde, pas un capteur solidaire du
> cap de la voiture. Un vrai jumeau n'aurait-il pas une caméra embarquée ? »

**Réponse.** Exact, et c'est précisément ce qui a causé notre bug de démo le plus
instructif : la caméra est un crop 128 px capté 60 px devant la voiture mais aligné sur
les axes du monde, pas sur le cap (pilot/signs.py:108-109). En diagonale, un panneau à
55 px d'offset latéral sortait du champ et perdait son premier chiffre, donc un « 30 »
amputé devenait un « 0 » que le modèle lisait 90 à confiance 1.00, en toute logique. Une
caméra solidaire du cap corrigerait ça côté simulateur ; on a préféré le régler côté
produit (offset 46 px = demi-crop 64 moins demi-panneau 18, visibilité garantie au
passage) parce que c'était la correction mesurée la plus sûre à une semaine de la
soutenance. C'est un raccourci assumé, pas un impensé.
**Porteur.** Charles (caméra) et Lorenzo (fix).
**Renvoi.** aucune (slide 9).

---

## B. Machine learning

### B1. Un IoU de 1,0000 ne prouve rien : problème trivial ou fuite de données ?
> « Un IoU parfait, en ML, n'impressionne personne : soit votre problème est trivial,
> soit vous avez une fuite entre train et validation. C'est lequel ? »

**Réponse.** Le premier, et on l'a prouvé plutôt que supposé : le seuillage de production
(`r>200 & g>200 & b>200`, perception.py:35) est au pixel près la fonction qui GÉNÈRE la
vérité terrain d'entraînement (sensors.py:125, même triple condition). L'IoU de 1,0
n'est donc ni un hasard ni une fuite, c'est une identité exacte : le U-Net a convergé
vers le prédicat que le domaine encode réellement, une piste blanche sur fond sombre. La
vraie leçon n'est pas « notre modèle est parfait » mais « on a MESURÉ que le ML était
superflu ici, et on a eu l'honnêteté de le remplacer par le seuillage équivalent » ; le
`.pth` entraîné reste dans le repo si le visuel se complexifie. Sur la fuite, le split
est par circuit (train gen_000-024, val gen_025-029), pas frame-à-frame.
**Porteur.** Nohlan.
**Renvoi.** A5.

### B2. Pourquoi pas de RL end-to-end ?
> « Pourquoi ne pas avoir fait du reinforcement learning end-to-end ? C'est l'approche
> naturelle pour de la conduite autonome, et vous êtes en filière IA. »

**Réponse.** Décision D1, actée en février avant la première ligne de code, pour trois
raisons qui ont toutes tenu. D'abord le critère de notation est explicitement la
généralisation sur circuit inconnu (« souvent la solution ne généralise pas », dixit le
sujet), or un RL pur surapprend le circuit d'entraînement et demande un temps qu'on
n'avait pas. Ensuite le pipeline perception vers contrôle se valide étage par étage (on
montre le masque, la trajectoire, les commandes) là où un RL est une boîte noire
indéfendable à l'oral. Enfin ça a payé : 30/30 sur des circuits jamais vus, avec un
compromis assumé (moins adaptatif dans les cas extrêmes du CNN, mitigé par le PID
backup). Ce n'est pas « le RL c'est mal », c'est « pas le bon outil pour CE critère ».
**Porteur.** Nohlan.
**Renvoi.** A1.

### B3. Votre CNN de conduite fait quelle taille ? (la doc dit ~300k)
> « Votre CNN de conduite, il pèse combien de paramètres ? Votre documentation annonce
> environ 300 000. »

**Réponse.** La docstring dit ~300k (cnn_drive_arch.py:22), le vrai chiffre est
**557 490**, et on le sait parce que l'audit du 09/07 l'a recalculé. Le point intéressant
n'est pas l'erreur de doc mais OÙ sont les paramètres : 524 416, soit 94 %, dans une
seule couche Flatten vers Linear(4096, 128) non régularisée, alors que les convolutions
ne pèsent que 4 %. C'est un vrai risque de surapprentissage sur le papier, qu'on a
documenté (Q8) au lieu de le cacher, avec le fix identifié (`AdaptiveAvgPool2d` à la
place du Flatten). On ne l'a pas corrigé parce qu'il ne s'est jamais matérialisé (30/30
constant, aucune régression) et qu'on a priorisé le fonctionnel démontré sur un risque
non observé, cohérent avec D9 : un risque assumé, pas un bug caché.
**Porteur.** Nohlan.
**Renvoi.** aucune.

### B4. Le CNN ne freine jamais (brake=0 codé en dur) : comment gère-t-il l'urgence ?
> « Votre CNN ne sort que volant et gaz, le frein est à zéro en dur. Comment votre voiture
> autonome s'arrête-t-elle en urgence ? »

**Réponse.** Le CNN produit `[steering, throttle]` et `cnn_policy` renvoie `brake=0.0` en
dur (control.py:161) : le freinage n'est pas dans le modèle appris, il est dans une
couche de règles séparée (emergency.py), par choix d'architecture. C'est cohérent avec le
behavioral cloning (le PID cloné ne freinait pas brutalement, il n'y avait rien à imiter)
et c'est un bon choix de sûreté : on ne veut pas qu'un modèle statistique non régularisé
soit responsable de l'arrêt d'urgence. La sécurité est donc déterministe : la règle
déclenche l'arrêt sur seuil dynamique `v²/(2·200)+38px`, zéro faux positif sur 40 571
frames, arrêt sans contact validé jusqu'à ~97 km/h. Un frein appris serait un chantier,
pas un progrès de sûreté.
**Porteur.** Nohlan (CNN) et Lorenzo (règle urgence).
**Renvoi.** aucune (slides 10-11).

### B5. Le CNN imite un PID que vous avez retuné depuis : il est périmé ?
> « Votre CNN clone un PID par imitation. Mais vous avez retuné ce PID en profondeur
> pendant l'optimisation. Le CNN conduit donc comme la version d'avant. Il sert à quoi ? »

**Réponse.** C'est exact et on l'assume : le CNN a cloné le PID d'avant optimisation
(throttle max 0.85, lidar_front_safe 250) alors que le PID actuel roule à throttle 1.0 et
lidar_front_safe 70 avec amortissement de sortie. Donc le CNN conduit proprement mais
« à l'ancienne », plus prudemment que le PID optimisé. Concrètement le CNN reste la
démonstration d'IA avancée (behavioral cloning fonctionnel, 30/30 en généralisation) et
le PID optimisé est ce qui court au chrono ; les deux coexistent par D9. Le
ré-entraînement du CNN sur le PID optimisé est un chantier M2 explicitement chiffré :
on ne l'a pas fait parce que le CNN ne concourt pas au temps de tour, pas par oubli.
**Porteur.** Nohlan.
**Renvoi.** aucune.

### B6. 95,13 % ou 90 % ? Vous dites que ça dépend de la seed.
> « Vous affichez 95,13 %, mais vous écrivez vous-même que selon la seed vous êtes entre
> 0,90 et 0,95. Alors votre modèle fait 90 ou 95 ? Lequel présentez-vous ? »

**Réponse.** On présente le checkpoint committé à 0,9513, reproductible bit à bit
(`best_val_acc=0.9513333`, epoch 46/100, dans `signs_cls_history.json`). La variance
0,90-0,95 vient de l'initialisation de la tête et de l'ordre des batches, variance
d'entraînement normale sur un petit jeu, qu'on a préféré documenter (annexe A3) plutôt
que taire. Deux garde-fous : le modèle livré est le MEILLEUR checkpoint, pas les derniers
poids (à l'epoch 100 la val retombe à 91,9 % pour un train à 98,3 %, surapprentissage
net, on s'est arrêtés au bon moment) ; et un seuil d'échec dur à 0,90 bascule sur GTSRB,
jamais déclenché. Réponse honnête : plancher 0,90, valeur livrée et reproductible 0,9513.
**Porteur.** Lorenzo.
**Renvoi.** A3.

---

## C. Méthodologie et mesure

### C1. Split « par circuit » : comment garantir zéro fuite train/validation ?
> « Vous dites split par circuit. Mais votre dataset panneaux réutilise des crops caméra
> des circuits d'entraînement. Comment être sûr qu'il n'y a aucune fuite ? »

**Réponse.** Le split est par circuit à tous les étages, et c'est vérifiable. Pour les
panneaux, les fonds sont de vrais crops caméra mais tirés en respectant la même
frontière : train = data/train (gen_000-024), val = data/val (gen_025-029), seed 42 pour
le train et 43 pour la validation (gen_signs_dataset.py:132-137) ; un circuit de
validation n'a jamais servi de fond à l'entraînement. Idem pour le CNN conduite et le
U-Net, validés sur les 5 circuits gen_025-029 jamais vus. La seule chose partagée est les
5 sprites de panneaux eux-mêmes, ce qui est voulu (ce sont les mêmes objets à
reconnaître) : la variabilité vient des fonds, angles, échelles et troncatures. Pas de
fuite frame-à-frame, qui serait la vraie faute.
**Porteur.** Nohlan (dataset conduite) et Lorenzo (dataset panneaux).
**Renvoi.** A5.

### C2. Dataset panneaux 100 % synthétique : que prouve 95 % du monde réel ?
> « Votre dataset panneaux est entièrement synthétique, cinq PNG augmentés. Qu'est-ce que
> 95 % sur des données que vous fabriquez prouve de la lecture réelle d'un panneau ? »

**Réponse.** Ça prouve exactement ce qu'on lui demande : la cible n'est PAS le monde
réel, c'est de lire les panneaux DE CE simulateur, qui sont eux-mêmes des sprites 2D ; il
n'y a pas de gap sim-to-real à franchir, le domaine de test EST le domaine de production.
Le 95 % est mesuré sur des fonds caméra réels du simulateur, sur des circuits de
validation jamais vus, avec 20 % de panneaux volontairement tronqués en augmentation.
On avait prévu un fallback GTSRB (50k images réelles) sous 0,90, jamais déclenché. La
limite honnête : ce classifieur ne saurait pas lire un panneau photographié dans la rue,
et ce n'était pas le sujet ; si le prof veut du réel, GTSRB est le plan B documenté (D7).
**Porteur.** Lorenzo.
**Renvoi.** A2 (et A3 pour GTSRB).

### C3. Votre « oracle » : qui l'a calculé, et l'avez-vous ajusté après coup ?
> « Vous dites rouler à 1-4 % du tour parfait. Mais qui a calculé cet oracle, et comment
> savons-nous que vous ne l'avez pas ajusté pour que vos résultats aient l'air bons ? »

**Réponse.** Justement parce qu'il a été calculé INDÉPENDAMMENT du résultat, par deux
voies séparées qui convergent. L'oracle vient d'une dérivation analytique (physique plus
contrôleur pure-pursuit) : 16,35 / 16,62 / 20,57 s sur les 3 circuits. Le PID optimisé,
réglé empiriquement par sweep de paramètres SANS regarder l'oracle, sort 16,45 / 16,95 /
21,28 s, soit +0,6 % / +2,0 % / +3,5 %. Deux méthodes complètement différentes qui
atterrissent à quelques pourcents l'une de l'autre : si l'oracle avait été bricolé après
coup pour flatter le PID, cette convergence croisée n'existerait pas. C'est cette double
dérivation, pas le chiffre seul, qui rend la borne crédible.
**Porteur.** Lorenzo.
**Renvoi.** A5.

### C4. Vos « cinq sous-agents spécialisés », c'est qui concrètement ?
> « Vous parlez d'une escouade de cinq sous-agents, un physicien, un ingénieur contrôle,
> un juriste... Ce sont qui, ces cinq personnes ? »

**Réponse.** Ce ne sont pas des personnes : ce sont cinq sous-agents Claude en lecture
seule, lancés en parallèle, chacun avec un périmètre de mesure étroit (physique,
contrôle, urgence, règles, perception), et on l'assume comme une méthode, pas comme un
tour de passe-passe. La valeur ajoutée humaine est à trois endroits : on a conçu le
protocole de mesure (le banc `bench_laps.py`, les métriques), on les a fait tourner en
LECTURE SEULE pour diagnostiquer sans rien casser, et surtout on a croisé leurs résultats
indépendants (l'oracle analytique de l'un contre le sweep empirique de l'autre,
convergence à quelques %) avant d'implémenter et committer nous-mêmes les correctifs.
Pour un projet d'IA, orchestrer des agents pour auditer 5607 lignes en 90 minutes est une
compétence, à condition de posséder et vérifier chaque résultat, ce qui est notre ligne.
**Porteur.** Lorenzo.
**Renvoi.** A6.

### C5. Zéro faux positif sur le jeu où vous avez réglé le seuil : overfit ?
> « Zéro faux positif sur 40 571 frames, c'est votre argument sécurité. Mais vous avez
> calibré le seuil sur ce même jeu. Ce n'est pas de l'overfitting à votre propre test ? »

**Réponse.** La critique est juste sur le principe, deux réponses. D'abord la règle n'est
pas un seuil ajusté à la main jusqu'à tomber sur zéro : c'est une loi physique (distance
d'arrêt `v²/2B` plus marge) plus un filtre géométrique (un mur est plat, une bordure de
virage est asymétrique, écart max 28 px entre les 3 rayons frontaux) ; 38 px et 28 px sont
les PLUS PETITES valeurs encore compatibles avec zéro FP, pas des valeurs larges choisies
pour être tranquilles. Ensuite le zéro est confirmé sur DEUX jeux distincts (dataset
synthétique négatif ET rejeu en conditions de production), et la sécurité positive
(arrêt AVANT le mur) est validée séparément par balayage de vitesse, zéro contact de 20 à
97 km/h. Ce que ça ne prouve pas : zéro FP sur un circuit jamais généré, ça on ne peut
pas l'affirmer.
**Porteur.** Lorenzo.
**Renvoi.** aucune (slide 10).

---

## D. Résultats et limites

### D1. Vous avez changé la physique pour vos panneaux : c'est vous arranger avec le réel ?
> « Vous avez monté MAX_ACCELERATION de 120 à 170 pour passer de 70,6 à 100 km/h, parce
> que sinon vos panneaux 90 ne servaient à rien. Déplacer le mur qui vous gêne, ce n'est
> pas un peu maquiller ? »

**Réponse.** On l'assume totalement, et on le dit sur la dernière slide plutôt que de le
cacher : c'est une décision PRODUIT explicite, pas une falsification de résultat. À
70,6 km/h un panneau « 90 » n'est jamais contraignant, donc la hiérarchie des limites
(30/50/90) n'est pas démontrable ; on a relevé le plafond pour que les trois limites
aient un effet visible. La distinction clé : on n'a maquillé AUCUN chiffre, on a changé
une constante de conception documentée ET tout revalidé à 100 km/h (test_signs 2/2, arrêt
au mur sans contact, CNN 30/30, STOP depuis 95,8 km/h). On est même transparents sur le
fait que 170 a été choisi pour retomber pile sur 100 (v* = 138,9 px/s) ; tricher aurait
été d'afficher 100 sans toucher la physique.
**Porteur.** Lorenzo (décision) et Charles (physique, proposition tracée `edd3e27`).
**Renvoi.** aucune (slide 13).

### D2. Le dashboard, c'est du debug : pourquoi 1,5 pt au barème ?
> « Votre dashboard avec ses courbes et son masque, c'est un outil de développement.
> Pourquoi ça devrait rapporter 1,5 point dans un projet de conduite autonome ? »

**Réponse.** Parce que le barème le prévoit explicitement comme livrable de visualisation
(A4) et parce que ce n'est pas que du debug : c'est la preuve visuelle en temps réel que
le pilote perçoit et décide ce qu'on affirme (caméra, masque segmenté, lidar, courbe de
vitesse avec ligne de limite, panneau détecté). Pour un jury, c'est ce qui distingue « la
voiture roule » de « voici exactement ce que la voiture voit et pourquoi elle freine »,
autrement dit de l'explicabilité. On ne le survend pas : il est validé à 54-58 FPS
dashboard compris, et l'esthétique pure (sprites, sons, traces) est comptée séparément
à +1 pt, on ne mélange pas les deux.
**Porteur.** Lorenzo.
**Renvoi.** A4.

### D3. Le CNN fait DNF sur gen_014 et n'est jamais mesuré en conditions réglementaires : trou ?
> « Votre CNN ne finit pas gen_014, et je ne vois nulle part de mesure du CNN avec
> panneaux et urgence actifs. C'est un angle mort, non ? »

**Réponse.** Les deux constats sont exacts et tracés dans nos données. Le CNN fait DNF sur
gen_014 : il roule mais ne croise jamais géométriquement la ligne d'arrivée (saut de
steering ~50°/frame à deux points récurrents), root-causé par l'agent B, reproduit à
l'identique dans `bench/physics100.json`, non corrigé. Et la config `cnn-regles` du banc
n'apparaît dans aucun des 4 JSON : le CNN n'a été mesuré qu'en isolation, jamais avec
panneaux plus urgence. La raison assumée : le CNN ne concourt pas au chrono (c'est le PID
optimisé), donc on a priorisé sa mesure en généralisation (30/30) ; le trou est réel,
documenté, et fait partie du chantier M2. On préfère le nommer que le laisser découvrir.
**Porteur.** Nohlan.
**Renvoi.** A5.

### D4. « 30/30, 0 % hors-piste » : et gen_026 et gen_013 alors ?
> « Vous affichez 30/30, 0 % hors-piste. Mais gen_026 sort à 0,4 %, et gen_013 était déjà
> à 0,4 % avant toute optimisation. C'est vraiment 30/30 à 0 % ? »

**Réponse.** C'est 30/30 au sens du critère (aucun circuit ne dépasse le seuil de 5 % de
hors-piste), et « 0 % » est un arrondi qu'on assume : gen_026 est à 0,4 %, sous le seuil.
gen_013 est plus intéressant : c'est le circuit le plus serré des 30 (rayon de courbure
23,1 px au 5e percentile) et il présentait déjà 0,4 % de hors-piste dans le baseline non
modifié ; on ne l'a pas planqué, notre réglage final ne l'aggrave pas (0,5 %) et un
réglage plus agressif fait même mieux (0,2 %). Formulation honnête : « 30/30 sous le seuil
de 5 %, dont deux circuits à quelques dixièmes de pour-cent, y compris le cas le plus dur
du lot qu'on n'a pas écarté ». Le 0 % de la slide est une simplification d'affichage.
**Porteur.** Nohlan.
**Renvoi.** aucune.

---

## E. Pièges et angles morts

### E1. Qui a fait quoi ? Nohlan, quelle partie du code est de vous ?
> « Nohlan, vous présentez le U-Net et le CNN de conduite. Concrètement, quelle partie du
> code est de vous ? On peut regarder le git log ensemble ? »

**Réponse (posture honnête à tenir).** La répartition documentée est Nohlan sur U-Net /
CNN conduite / dataset conduite, Lorenzo sur panneaux / optimisation, Charles sur le
simulateur. À l'ouverture du git, deux réalités à NE PAS masquer : l'historique ne montre
que deux identités auteur (Lorenzo ~45 commits, Charles ~7), aucun commit signé Nohlan ;
le travail ML a transité par la machine et l'identité de Lorenzo (pair-programming,
commits groupés). La règle d'or : ne JAMAIS revendiquer nominativement des commits qui
n'existent pas. Nohlan doit défendre le fond technique du U-Net et du CNN (c'est lui qui
porte les slides 5-7) et présenter sa contribution comme conception / design /
pair-programming, honnêtement. Si le jury projette `git log`, mieux vaut l'avoir annoncé
que se faire prendre.
**Porteur.** Nohlan (appui Lorenzo sur la partie historique git).
**Renvoi.** aucune.

### E2. Sécurité mur validée à 200 px, livrée à 130 px : revalidée ?
> « Vos rapports valident la sécurité du mur, zéro collision, avec une apparition à 200 px.
> Le code livré met le mur à 130 px. Avez-vous revalidé à la distance que vous livrez ? »

**Réponse.** Non, et c'est la réponse honnête : la validation « zéro collision » a été
faite avec `SPAWN_DISTANCE=200 px`, alors que la valeur livrée est 130 px (wall.py:29,
changée le 10/07 à 9h58, après les rapports écrits dans la nuit). On ne l'a pas rejouée à
130. Deux raisons de ne pas paniquer : 130 px reste dans la plage où l'arrêt a été validé,
et la marge de sécurité était de 1,6× à 2,75× le plafond physique, on ne la consomme pas
en passant de 200 à 130 ; et il y a un filet indépendant du réglage (au contact, le
simulateur téléporte la voiture et annule sa vitesse, main.py), donc même un échec de
freinage ne produit pas de traversée de mur. Rejouer le balayage à 130 est un test d'une
heure qu'on aurait dû faire : un manque, pas un risque avéré.
**Porteur.** Charles (constante mur) et Lorenzo (validation urgence).
**Renvoi.** aucune.

### E3. « IA avancée » à 2 pts : U-Net remplacé par un seuil, MobileNet gelé. Où est l'IA ?
> « Vous réclamez 2 points d'IA avancée. Mais votre U-Net est remplacé par un seuillage et
> votre classifieur gèle MobileNet en n'entraînant qu'une petite tête. Où est l'IA avancée ? »

**Réponse.** L'IA avancée revendiquée, c'est le CNN de conduite par behavioral cloning :
un réseau qui apprend une politique de contrôle (masque 64×64 plus lidar vers volant plus
gaz, MAE 1,80°) et généralise à 30 circuits jamais vus. Le U-Net remplacé par un seuillage
n'est pas un aveu de faiblesse mais une démarche mesurée (l'IoU a PROUVÉ que le ML était
superflu ici avant de simplifier), et le transfer learning gelé sur les panneaux est une
technique standard et appropriée pour un petit dataset, pas un raccourci. Le vrai argument
n'est pas le nombre de paramètres entraînés mais la chaîne : perception apprise, contrôle
appris, généralisation démontrée sur données jamais vues, avec un PID baseline pour
prouver que le CNN apporte quelque chose. Si le jury juge que ça ne vaut pas 2 pts, c'est
un débat de barème légitime, mais ce n'est pas de l'IA facile.
**Porteur.** Nohlan (CNN/U-Net) et Lorenzo (panneaux).
**Renvoi.** A4.

---

## Synthèse stratégique

### Les 3 questions les PLUS probables (plan de réponse en 3 temps)

**MP-1 (= B1) : « IoU 1,0, vous l'avez remplacé par un seuillage : où est le ML ? »**
La slide 5 est un appât explicite ; un prof IA mord dessus quasi certainement.
1. Cadrer sans se défendre : oui IoU 1,0, et c'est une IDENTITÉ exacte (le seuillage de
   prod est la fonction qui génère le GT au pixel près), pas une fuite ni un hasard.
2. Retourner en maturité : on l'a MESURÉ avant de simplifier ; converger vers le plus
   simple prouve la bonne invariance apprise ; le `.pth` reste prêt si le visuel change.
3. Verrouiller la généralisation : split par circuit (gen_000-024 / gen_025-029), pas de
   fuite frame-à-frame ; le vrai ML avancé est le CNN de conduite, pas le U-Net.

**MP-2 (= C1) : « Comment prouvez-vous que ça généralise, sans fuite de données ? »**
C'est le critère n°1 énoncé par le prof lui-même (slide 7 tourne autour de sa citation).
1. Rappeler la contrainte comme un choix pris tôt (D10, avril), pas subi : split par
   circuit à tous les étages, augmentations, aucun circuit de test vu à l'entraînement.
2. Donner la preuve chiffrée : 30/30 sur les 5 circuits jamais vus, 0 % hors-piste (à
   deux circuits près sous le seuil), classifieur 95,13 % sur fonds de circuits val.
3. Désamorcer la fuite panneaux : les fonds val viennent de gen_025-029, seed 43 distincte,
   jamais utilisés à l'entraînement ; seuls les 5 sprites sont partagés, à dessein.

**MP-3 (= D1) : « 70,6 puis 100 km/h pour vos panneaux, c'est pas vous arranger ? »**
La punchline de la slide 13 (« c'est le mur qu'on a déplacé ») invite littéralement le piège.
1. Assumer d'emblée : décision produit explicite, DITE sur la slide, jamais cachée.
2. Distinguer maquiller un chiffre (jamais) de déplacer un paramètre documenté (fait) :
   tout revalidé à 100 (test_signs 2/2, mur sans contact, CNN 30/30, STOP à 95,8 km/h).
3. Justifier le pourquoi produit : à 70,6, un « 90 » n'est jamais contraignant, donc la
   hiérarchie des limites n'est pas démontrable ; transparence sur le 170 choisi pour 100 pile.

*(4e très anticipée, annexe A1 déjà prête : « pourquoi pas de RL ». Traiter comme MP mais
risque faible car préparée : dérouler D1, 30/30, compromis PID backup.)*

### Les 2 questions les PLUS dangereuses (piège explicité)

**MD-1 (= E1) : « Qui a codé quoi ? Nohlan, montrez-nous dans le git log. »**
*Piège :* sur-revendiquer une contribution individuelle que le git ne montre pas. Nohlan
présente le U-Net et le CNN mais n'a AUCUN commit (2 auteurs seulement : Lorenzo ~45,
Charles ~7). Si quelqu'un affirme « Nohlan a codé le U-Net » et que le jury projette
`git log`, la crédibilité de TOUTE l'équipe s'effondre en direct, et ça touche la note
individuelle. *Désamorçage :* anticiper, parler de rôles (conception, pair-programming)
sans inventer de commits, et s'assurer que Nohlan maîtrise techniquement ses slides 5-7.

**MD-2 (= A3) : « Physique framerate-dépendante, banc à dt fixe : vos chiffres = la démo ? »**
*Piège :* prétendre que le banc décrit la démo, OU nier la dépendance au framerate, qui
est dans le code (`car.speed -= car.speed * FRICTION`, physics.py:60), donc vérifiable en
direct. Un prof d'archi peut alors démonter TOUTE la narrative « on ne règle pas ce qu'on
ne mesure pas » (slides 10-11, où se logent 3,5 pts). *Désamorçage :* assumer le dt fixe
comme choix de déterminisme, donner les FPS réels de la démo (54-58, même régime que le
banc), citer `friction^dt` comme chantier M2 déjà identifié. Ne jamais esquiver.

*(Dangers de second rang à surveiller : C4 « vos 5 agents = des IA ? » (assumer comme
méthode, ne pas laisser croire à des humains) ; C5 « zéro FP overfit au test ? » (border
la garantie) ; E2 SPAWN_DISTANCE 130 non revalidé.)*

### Les 5 « ne dites JAMAIS » devant ce jury

1. **« C'était un faux positif de l'audit »** (à propos du DriveCNN 557k / Q8). C'est un
   VRAI écart doc/archi et un vrai risque de surapprentissage. Dire : « un risque réel,
   documenté et assumé, qui ne s'est jamais matérialisé », pas « faux positif ».
2. **« Le U-Net n'a servi à rien / on l'a raté. »** L'IoU a PROUVÉ que le seuillage
   suffisait ici. Converger vers le plus simple est une réussite, pas un échec ; le
   `.pth` reste prêt. Vendre la simplification comme maturité, jamais comme aveu.
3. **Attribuer nominativement du code que le git ne montre pas** (« Nohlan a codé le
   U-Net et le CNN »). Parler de rôles et de pair-programming, sans revendiquer de
   commits inexistants. C'est le piège le plus coûteux si `git log` est projeté.
4. **« Le freinage d'urgence est sûr / infaillible / zéro faux positif »** en point final.
   Toujours border : « zéro FP mesuré sur 40 571 frames de deux jeux, plus arrêt sans
   contact validé jusqu'à ~97 km/h ». Une mesure, jamais une garantie universelle.
5. **« La voiture tient 50,0 km/h pile / on ne dépasse jamais la limite »** comme un
   contrôle strict. C'est un équilibre gaz réduit (0,15) contre friction ; le frein actif
   ne s'enclenche qu'à limite plus 15 km/h. La voiture FLOTTE sous la limite, elle n'est
   pas clampée. Ne pas vendre une précision qu'on n'a pas.
