# TRAME ORALE : Soutenance Autonomous Twin (M1 ESGI)

Document de travail pour Charles, Nohlan et Lorenzo. Compagnon oral du deck
`index.html` : le script complète les slides, il ne les lit pas. Version du 21/07/2026.

## En-tête de cadrage

- **Format** : 15:00 d'oral + 15:00 de questions, 3 orateurs, démo live incluse.
- **Durée totale cible : 13:20** (marge 1:40 sur les 15:00 accordées — fusion des
  anciennes S10 et S11 le 23/07, le passage de Lorenzo gagne 40 secondes).
- **Répartition** : Charles S1 à S4 (~4:00), Nohlan S5 à S7 (~3:30), Lorenzo S8 à S10 (~3:20),
  démo S11 (~1:50, équipe), conclusion S12 (~0:40, Lorenzo).
- La démo peut reprendre jusqu'à 2:30 en mangeant la marge, jamais au delà.

| Slide | Titre | Orateur | Durée | Cumul |
|---|---|---|---|---|
| S1 | Titre | Charles | 0:30 | 0:30 |
| S2 | Le concept | Charles | 1:00 | 1:30 |
| S3 | Architecture | Charles | 1:15 | 2:45 |
| S4 | Le monde | Charles | 1:15 | **4:00** |
| S5 | Percevoir | Nohlan | 1:10 | 5:10 |
| S6 | Conduire | Nohlan | 1:05 | 6:15 |
| S7 | Généraliser | Nohlan | 1:15 | **7:30** |
| S8 | Lire les panneaux | Lorenzo | 1:00 | 8:30 |
| S9 | Le bug qui n'en était pas un | Lorenzo | 1:10 | 9:40 |
| S10 | Résultats (fusion) | Lorenzo | 1:10 | **10:50** |
| S11 | Démo live | Équipe | 1:50 | 12:40 |
| S12 | Conclusion | Lorenzo | 0:40 | **13:20** |

**Conventions de lecture de ce document**

- Le texte courant du script se dit quasi mot à mot ; adaptez le mot, jamais le fond ni les chiffres.
- Les passages en **gras entre guillemets** sont des verbatims : à dire exactement tels quels.
- Les [crochets] sont des indications scéniques : elles ne se prononcent pas.
- On ne lit pas les slides. Deux exceptions volontaires, parce que ce sont des citations :
  la phrase du sujet (S7) et la phrase finale (S12).
- L'orateur qui vient de finir tient le chrono du suivant. Points de contrôle en fin de document.

---

## S1. Titre (Charles, 0:30, cumul 0:30)

[regarder le jury, pas l'écran]

Bonjour. Nous sommes Charles, Nohlan et Lorenzo, et nous vous présentons Autonomous Twin,
notre jumeau numérique automobile. L'idée tient en une phrase : un simulateur qui fait
rouler une voiture, et une IA qui la conduit sans jamais voir autre chose que ses capteurs.

Je pose le décor. Nohlan vous montrera comment l'IA apprend à conduire. Lorenzo, comment
elle lit les panneaux, et comment on a tout mesuré. Et on termine par une démonstration
en direct.

**Transition** : Commençons par le concept.

---

## S2. Le concept (Charles, 1:00, cumul 1:30)

Le projet, ce sont deux programmes qui tournent en même temps et qui ne partagent rien.
D'un côté, le simulateur : c'est lui qui possède la vérité. La physique, le circuit, le
chronomètre, les panneaux, les murs.

De l'autre, le pilote. Et le mot sur la slide n'est pas une image : il est réellement
aveugle. Ce processus n'a physiquement accès à rien d'autre que ce qui passe sur le
réseau. Une image caméra de 128 pixels de côté, cinq rayons lidar, sa vitesse. C'est
tout. Pas de position, pas de plan du circuit, aucun accès à l'état interne du simulateur.

[pause]

Cette contrainte, gardez-la en tête : c'est elle qui va structurer toutes les décisions
de l'équipe. Y compris la façon dont on a collecté nos données d'entraînement, vous le
verrez : tout passe par la même prise.

**Transition** : Et cette prise, justement, la voici.

---

## S3. Architecture (Charles, 1:15, cumul 2:45)

[pointer le schéma]

Concrètement, le simulateur est un serveur ZeroMQ, le pilote un client. Une requête, une
réponse : le simulateur envoie les capteurs, le pilote renvoie ses trois commandes,
volant, gaz, frein. Soixante fois par seconde. Ce mode synchrone nous garantit zéro
dérive d'horloge entre les deux processus.

Ce découplage a deux bénéfices très concrets. D'abord, on a développé le simulateur et le
pilote en parallèle pendant tout le semestre, sans jamais se bloquer. Ensuite, l'équipe
ML a pu brancher son enregistreur de données directement sur ce flux, sans dépendre de ma
branche.

[pause]

Et une anecdote qui montre que ce genre d'architecture se règle dans le réel. Le jour où
le premier réseau de neurones est entré dans la boucle, le timeout réseau était réglé à
100 millisecondes. Largement suffisant pour un contrôleur classique. Mais le tout premier
calcul d'un modèle PyTorch à froid prend plus que ça. Trente-deux minutes après le commit
du CNN, on passait le timeout à cinq secondes. Dès qu'on met du ML dans une boucle temps
réel, les hypothèses d'infrastructure craquent.

**Transition** : Voyons maintenant ce que ce serveur fait vivre.

---

## S4. Le monde (Charles, 1:15, cumul 4:00)

Ce monde n'est pas un décor : il vit. La physique, c'est le modèle bicycle du sujet,
quatre équations, avec un plafond de vitesse calibré à 100 kilomètres heure. Lorenzo vous
racontera en conclusion pourquoi ce chiffre est une décision, pas un hasard.

Le mur dynamique, lui, apparaît devant la voiture, visible du lidar et de la caméra en
même temps. Et retenez ce détail : s'il n'est pas atteint, il expire au bout de six
secondes. Ça paraît anodin, ça deviendra important au moment des résultats.

[pause]

Le chronomètre, ensuite. Compter un tour, ça a l'air simple, mais un vrai compteur de
tours doit résister aux allers-retours sur la ligne. Le nôtre ne compte un passage que si
la voiture s'est d'abord vraiment éloignée : un système d'armement, pas un simple capteur.

Enfin, les panneaux sont décrits dans un fichier posé à côté de chaque circuit, puis
composés directement dans l'image caméra du pilote. Et pour le plaisir : un vrai son
moteur, des traces de pneus, un compteur à cadran. Vous verrez tout ça en démo.

**Transition, relais nommé** : Voilà pour le monde. Nohlan, je te laisse mettre une
intelligence au volant.

---

## S5. Percevoir (Nohlan, 1:10, cumul 5:10)

Merci Charles. Première brique du pilote : voir la route. On a entraîné un U-Net de
zéro : il prend l'image caméra et sort un masque, route ou pas route. Résultat sur cinq
circuits jamais vus à l'entraînement : une IoU de 1,0.

[pause] [regarder le jury]

Pas 0,99. 1,0. Et un score parfait, en ML, ce n'est pas une victoire : c'est un signal
d'alarme. L'écart entre la prédiction et la vérité terrain était exactement zéro. Alors
on a cherché ce que le réseau avait vraiment appris. Verdict : un seuillage de luminance.
La piste est claire sur fond sombre, le problème est objectivement soluble par un seuil.

Donc on a fait la chose honnête : on a remplacé le U-Net par le seuillage. Strictement
équivalent, zéro coût de calcul. Le modèle entraîné reste dans le repo, prêt à reprendre
du service si le visuel se complexifie un jour.

Ce n'est pas un échec du ML. C'est la preuve qu'on a vérifié ce que le réseau apprenait,
au lieu de s'admirer.

**Transition** : Voir la route, c'est fait. Maintenant, la conduire.

---

## S6. Conduire (Nohlan, 1:05, cumul 6:15)

Pour conduire, on a fait un choix de méthode avant un choix de modèle : d'abord un pilote
classique, un PID, qui boucle des tours propres. Fonctionnel d'abord, ML ensuite. Ce PID,
c'est deux choses : notre baromètre, pour juger si le réseau apporte vraiment quelque
chose, et notre filet de sécurité, disponible à tout moment.

Ensuite seulement, le CNN. Du behavioral cloning : il apprend en imitant les trajectoires
du PID. Et là, LA décision du projet : en entrée, ce CNN ne reçoit jamais l'image caméra.
Il reçoit le masque de la slide précédente, plus le lidar. Autrement dit, il n'apprend
que la forme de la route, jamais son apparence. Changez les couleurs, le style, le thème
du circuit : il s'en moque.

[pointer le compteur]

Le résultat, c'est ce chiffre : trente circuits sur trente, sans sortir de piste, sur des
circuits qu'il n'avait jamais vus.

**Transition** : Trente circuits jamais vus. Encore fallait-il les avoir. C'était la
consigne numéro un du sujet.

---

## S7. Généraliser (Nohlan, 1:15, cumul 7:30)

Cette phrase à l'écran vient du sujet lui-même, et on l'a prise comme un avertissement
personnel. Je la cite : **« souvent la solution ne généralise pas et ne permet pas
d'exécuter la conduite sur différents circuits »**.

[pause]

Autrement dit, le piège classique, c'est une IA qui récite son circuit d'entraînement.
On a donc mis cette contrainte dans le protocole dès avril : le découpage entre
entraînement et validation se fait par circuit, jamais par image. Aucun chiffre de cette
présentation n'est mesuré sur du déjà-vu.

Restait à avoir des circuits. On a hésité entre deux options : un monde infini qui
défile, qui demandait une refonte complète de la physique, trois à cinq jours de travail,
ou un générateur procédural de circuits, un après-midi. On a choisi le générateur.

[pointer la galerie]

Et il a fallu itérer : la première version ne produisait que des patates rondes, trop
uniformes. La version deux contrôle la courbure explicitement : de vraies épingles, de
vraies chicanes, de vraies lignes droites. Trente circuits, reproductibles par seed :
vingt-cinq pour l'entraînement, cinq pour la validation.

**Transition, relais nommé** : Et sur ces circuits, il n'y a pas que de la route. Il y a
des règles. Lorenzo, à toi.

---

## S8. Lire les panneaux (Lorenzo, 1:00, cumul 8:30)

Merci Nohlan. Mon acte commence le 9 juillet. Le projet a dormi deux mois et demi, zéro
commit, toute l'équipe. Je reviens avec, honnêtement, **« plus rien en tête »**. Alors
avant de recoder quoi que ce soit, on a audité l'existant : cinq revues en parallèle, en
lecture seule. Résultat : cinq trous concrets, documentés. Et c'est ce qui a permis de
livrer toute la lecture de panneaux le soir même.

[pointer le pipeline]

La chaîne fait quatre étages : détecter le panneau par sa couleur, le classifier avec un
MobileNetV2 dont seule la tête est réentraînée, confirmer sur trois images pour éviter
les faux positifs, puis obéir : ralentir à 30, 50 ou 90, marquer le stop.

Ce que la ligne du bas ne dit pas, c'est la méthode. À 0,83 de précision, le diagnostic
montrait que le problème était la résolution, pas le manque de données. Résolution 224,
puis une vraie tête MLP : 95 pour cent en validation, sur des fonds caméra réels.

**Transition** : Sauf qu'un soir de démo, ce classifieur à 95 pour cent nous a sorti une
absurdité.

---

## S9. Le bug qui n'en était pas un (Lorenzo, 1:10, cumul 9:40)

Un panneau 30, lu 90, avec une confiance de 100 pour cent. En pleine démo.

[pause]

Le réflexe, c'est d'accuser le modèle. On a aligné les hypothèses ML : marge trop fine du
classifieur ? Hystérésis à durcir ? Toutes plausibles. Toutes fausses. Au lieu de
choisir, on a instrumenté : une sonde qui enregistre, image par image, ce que le
classifieur reçoit vraiment. Et là, tout s'éclaire.

[pointer l'animation]

La caméra n'est pas accrochée au nez de la voiture : c'est un cadre aligné sur le monde.
À 55 pixels de décalage, le panneau sort du champ par le bord, et le 3 disparaît en
premier. Le modèle recevait un zéro amputé. Et un 0 tout seul, c'est la fin de 50 comme
de 90 : le modèle répondait juste.

[pause]

Le correctif est géométrique, pas ML : placement recalculé à 46 pixels pour garantir la
visibilité, filtre qui rejette les panneaux tronqués, recentrage du crop, seuils
recalibrés. Confiance remontée à 0,94. **« C'est la sonde frame par frame qui a tranché,
pas le raisonnement. »** **« Le bug ML n'était pas dans le ML. »**

**Transition** : Cette leçon, on l'a industrialisée : ne plus jamais débattre quand on
peut mesurer.

---

## S10. Résultats (Lorenzo, 1:10, cumul 10:50) — fusion des anciennes S10 et S11 (23/07)

Au moment de préparer la démo, le constat, c'était : **« la voiture est hyper
lente »**. Avant de bricoler, on a construit de quoi mesurer : un banc. De vrais tours
comptés au franchissement de ligne, déterministe, tout archivé. Et le banc a parlé :
le vrai coupable, c'était le freinage d'urgence, déclenché sans aucun mur dans les
virages serrés. Au point que ce circuit-là ne finissait jamais ses trois tours.
**« C'est le banc qui a tranché »**, pas l'intuition.

[pointer les quatre cartes, une par une]

Le traitement tient en quatre corrections, mesurées une à une. Le freinage fantôme
corrigé : la voiture ne freine plus que si un impact est réellement possible. Le
pilote débridé. La perception accélérée. Et les règles précisées, avec l'arrêt au
stop à 0,8 seconde.

[pointer les graphes]

Résultat, à règles constantes, sur les mêmes circuits : moins 54 à moins 60 pour cent,
et le circuit qui ne finissait jamais boucle ses trois tours. Le freinage parasite
passe de 15 pour cent du temps à 0,8. Et pour situer tout ça, on a calculé un oracle,
le tour parfait théorique : en configuration libre, notre pilote roule à 1 à 4 pour
cent de ce temps parfait.

**Transition, relais nommé** : Assez de chiffres. On vous montre. Charles, Nohlan, la
démo.

---

## S11. Démo live (Équipe, 1:50, cumul 12:40 ; plafond absolu 2:30)

[bascule vers le simulateur, lancé avant la soutenance et laissé en pause]
[rôles : Charles au clavier, pose le mur avec Espace ; Lorenzo commente le dashboard ;
Nohlan annonce les étapes et tient le chrono]

**Nohlan, cadrage (10 s)** : Tout est en direct. Le circuit s'appelle gen_014. Retenez ce
nom : c'est précisément celui qui ne finissait pas ses trois tours avant l'optimisation.

**Étape 1, zone 50 (25 s), Lorenzo** : La voiture est en zone limitée : régulation à
50,0, pile sous la limite. [laisser rouler quelques secondes en silence]

**Étape 2, stop (20 s)** : Nohlan : Panneau stop. Lorenzo : Arrivée à 95 kilomètres
heure, arrêt complet marqué 0,8 seconde, et ça repart.

**Étape 3, mur (25 s)** : Charles : Je pose un mur, maintenant. [appui sur Espace]
Lorenzo : Arrêt net, zéro contact. Et à basse vitesse, ce mur aurait expiré avant
l'impact : pas de risque réel, pas de freinage. C'est le comportement dont je vous
parlais.

**Étape 4, épingle (20 s)** : Nohlan : L'épingle, maintenant, celle qui déclenchait le
freinage fantôme. Lorenzo : 70 kilomètres heure, braquage complet, aucun déclenchement.

**Étape 5, dashboard (15 s), Lorenzo** : Et tout du long, le dashboard : caméra, masque,
lidar, courbe de vitesse. C'est tout ce que le pilote sait du monde.

**Nohlan, clôture et relais nommé (5 s)** : Fin de la démo. Lorenzo, pour conclure.
[revenir au deck, slide 12]

**Si un process tombe** : Charles relance les deux, une dizaine de secondes, procédure
éprouvée. Pendant ce temps, Nohlan meuble avec le circuit : gen_014, son histoire de DNF,
sa correction. On ne s'excuse pas, on ne commente pas la panne.

---

## S12. Conclusion (Lorenzo, 0:40, cumul 13:20)

[pause] [regarder le jury]

**« Nous savons pourquoi nous ne pouvons pas aller plus vite. »**

[pause]

Ce plafond n'était ni un bug, ni un manque de temps. Il était physique, démontré par
l'oracle : 70,6 kilomètres heure, l'équilibre exact entre l'accélération et le
frottement. Et une fois qu'on l'a su, on a fait un choix de produit, parce que sans
plafond relevé, un panneau 90 ne contraint rien : **« quand le mur gênait le produit,
c'est le mur qu'on a déplacé »**. De 70,6 à 100, en connaissance de cause.

La suite est à l'écran, déjà chiffrée. Merci : place à vos questions.

---

# Réserves

Ce qu'on dit SI le temps le permet, à l'endroit indiqué. Décision aux points de contrôle
chrono (voir plan de répétition). Ne jamais improviser une réserve en retard.

**R1. DriveCNN : 557 000 paramètres réels (20 s). Où : S6, juste après « trente sur
trente », par Nohlan. Sinon : munition Q&R.**
Petite confession d'architecte au passage : la documentation du modèle annonçait environ
300 000 paramètres. Notre audit a refait le calcul : 557 000, dont 94 pour cent
concentrés dans une seule couche. C'est le symétrique exact de l'histoire du U-Net : on
annonce un chiffre, on va vérifier dans le code. Risque de surapprentissage identifié,
documenté, assumé : il ne s'est jamais matérialisé, trente sur trente constant.
Attention à la formulation : c'est un risque réel qu'on assume, pas un faux positif de
l'audit, et pas un bug caché.

**R2. La journée fondatrice du 24 avril (15 s). Où : fin de S7, par Nohlan.**
D'ailleurs, tout ce que je viens de raconter, le PID, le U-Net, le CNN, le générateur de
circuits, tient en une seule journée de travail, le 24 avril. Le journal du projet la
résume en une ligne : pipeline ML complet en une journée.

**R3. Le seuillage est identique au pixel près (10 s). Où : S5, après « strictement
équivalent », par Nohlan.**
Et quand on dit équivalent : le seuillage en production est, au pixel près, la même
condition que celle qui générait la vérité terrain. Ce n'est pas une approximation, c'est
une identité.

**R4. Deux méthodes indépendantes convergent sur l'oracle (15 s). Où : S11, après « 1 à 4
pour cent », par Lorenzo.**
Et cet oracle n'a pas été calculé pour coller au résultat : un balayage empirique de
paramètres d'un côté, une dérivation physique de l'autre, menés séparément, convergent à
quelques pour cent l'un de l'autre.

**R5. Les arrêts au stop n'ont pas bougé (10 s). Où : S11, après les barres, par
Lorenzo.**
Précision importante : le temps passé à l'arrêt aux stops est identique avant et après.
Tout le gain vient du gaspillage éliminé, pas des règles contournées.

**R6. La vitesse à plafond 100 (10 s). Où : S12, après « en connaissance de cause », par
Lorenzo.**
Depuis, en configuration libre, la voiture croise à 97,6 kilomètres heure de moyenne, et
le meilleur tour est passé de 16 secondes et demie à 11 secondes 57.

---

# Renvois questions

Si le jury pose la question PENDANT l'oral : une phrase de réponse, puis renvoi explicite
« le détail est en annexe, on y revient pendant les questions », et on reprend le fil. Ne
jamais dériver plus de 20 secondes hors du script.

1. **Pourquoi pas du RL end-to-end ?** Parce que le critère de notation était
   **« 3 tours sur circuit inconnu »** : un RL apprend par cœur son circuit, notre
   pipeline ne voit que la forme de la route. Détail en annexe A1.
2. **Quelle taille fait votre modèle de conduite ?** 557 000 paramètres réels là où la
   doc disait 300 000, écart trouvé par notre propre audit, risque documenté et assumé,
   jamais observé en test. Détail en annexe A6 (et réserve R1).
3. **Le mur ne trompe jamais le détecteur de panneaux ?** Non : le rouge brique reste
   sous le seuil du détecteur, vérifié en boucle fermée. Détail en annexe A2.
4. **Pourquoi 95 pour cent et pas plus, après 100 époques ?** Parce qu'on livre le
   meilleur checkpoint, l'époque 46 : à l'époque 100, la validation était retombée à 92
   malgré un train à 98, surapprentissage net. Détail en annexe A3.
5. **Votre pire confusion ?** La classe 90 : rappel 89,7 pour cent, confondue surtout
   avec 50 ; la matrice complète est dans le repo. Détail en annexe A3.
6. **Le CNN profite-t-il de vos optimisations ?** Pas encore : il clone le PID d'avant
   optimisation, son réentraînement est le premier chantier annoncé en conclusion, et le
   seul cas d'échec restant est déjà diagnostiqué et reproductible. Détail en annexe A5.
7. **Pourquoi l'image saccade quand un panneau apparaît ?** La classification coûte
   93 millisecondes CPU par image, le débit instantané tombe à 9 à 11 images par seconde
   pendant une rafale : mesuré, documenté, sans effet sur la sécurité. Détail en
   annexe A5.
8. **Comment tenez-vous 50,0 pile ?** Ce n'est pas un blocage logiciel : un équilibre
   entre gaz réduit et frottement, le frein actif n'intervient qu'au delà de la limite
   plus 15. Réponse directe, pas d'annexe.
9. **Et si on pose le mur à basse vitesse ?** Il expire au bout de six secondes : sous
   48 kilomètres heure il disparaît avant l'impact, pas de risque réel donc pas de
   freinage, c'est le freinage par le risque. Réponse directe.
10. **Avez-vous revalidé l'arrêt au mur après le changement de distance d'apparition ?**
    Réponse honnête : la campagne de validation a tourné avec une distance plus généreuse
    que la valeur livrée, elle n'a pas été rejouée depuis ; le simulateur garde un
    garde-fou physique indépendant, et la démo vient de le montrer en conditions réelles.
11. **Comment vous êtes-vous coordonnés à trois ?** On a un contre-exemple honnête : deux
    d'entre nous ont recodé le même dashboard, faute de signal explicite « je prends le
    sujet » ; depuis, chaque chantier est annoncé avant d'être ouvert. Détail en
    annexe A6.
12. **Qui a fait quoi ?** Point à caler à trois AVANT la répétition : l'historique git ne
    montre que deux auteurs de commits ; préparer ensemble la réponse sur la répartition
    réelle (binômes, revue, données, slides, démo) plutôt que la découvrir en direct.

---

# Plan de répétition

**Modalités** : deux répétitions chronométrées à trois minimum avant le jour J, dont une
avec la vraie démo. L'orateur qui vient de finir tient le chrono du suivant ; pendant la
démo, c'est Nohlan. On répète aussi les transitions de relais : c'est là que le temps se
perd.

**Trois points de contrôle chrono**

1. **Fin S4 : cible 4:00.** Nohlan regarde le chrono en prenant le relais.
   Si 4:20 ou plus : coupe C1 dans S7 (réduire le choix monde infini contre générateur à
   une seule phrase, gain 15 s), et aucune réserve avant S8.
   Si 3:45 ou moins : réserves R2 et R3 autorisées.
2. **Fin S7 : cible 7:30.** Lorenzo regarde le chrono en prenant le relais.
   Si 7:50 ou plus : coupe C2 dans S10 (supprimer la phrase sur l'ancienne évaluation
   10 à 17 pour cent, gain 10 s) et coupe C3 dans S11 (supprimer la phrase « environ
   90 minutes », gain 8 s).
   Si 7:15 ou moins : réserves R4 et R5 autorisées.
3. **Fin S10 : cible 10:50.** Nohlan annonce discrètement le temps au moment de la
   bascule.
   Si 12:00 ou plus : démo en version courte, étapes 1, 3 et 5 uniquement (limite, mur,
   dashboard), environ 1:15.
   Si 11:15 ou moins : démo complète tranquille, réserve R6 autorisée en conclusion.

**Règle générale de coupe** : on coupe des phrases de contexte, jamais les verbatims,
jamais les chiffres du barème, jamais les transitions de relais. La conclusion (S12)
n'est jamais coupée : les 40 secondes finales sont sanctuarisées.
