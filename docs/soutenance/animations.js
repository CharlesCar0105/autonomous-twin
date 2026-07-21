/* Animations du deck — anime.js v4 (UMD, global `anime`) déclenchées par Reveal.
   Chaque fabrique reçoit la <section> et retourne une timeline ; elle est
   revert() à la sortie de slide pour que l'animation se rejoue à chaque entrée. */

const { animate, createTimeline, stagger, utils } = anime;

/* Rendu statique : export PDF (?print-pdf), ou préférence système
   reduced-motion. Dans ces cas on n'anime pas — on pose les états finaux. */
const STATIQUE = /print-pdf/.test(location.search)
  || (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);

/* Découpe le titre en lettres une seule fois (pour l'anim d'intro). */
const titre = document.getElementById('titre-principal');
titre.innerHTML = titre.textContent.replace(/\S/g, '<span>$&</span>');

const ANIMS = {

  intro(slide) {
    return createTimeline()
      .add('#titre-principal span', {
        translateY: ['0.8em', 0], opacity: [0, 1],
        delay: stagger(45), duration: 700, ease: 'outExpo',
      })
      .add('.soustitre', { opacity: [0, 1], duration: 600 }, '-=300')
      .add('.equipe', { opacity: [0, 1], duration: 600 }, '-=200');
  },

  concept(slide) {
    return createTimeline()
      .add(slide.querySelector('.carte-simu'), {
        translateX: ['180px', 0], opacity: [0, 1], duration: 800, ease: 'outExpo',
      })
      .add(slide.querySelector('.carte-pilote'), {
        translateX: ['-180px', 0], opacity: [0, 1], duration: 800, ease: 'outExpo',
      }, '-=800')
      .add(slide.querySelector('.punchline'), { opacity: [0, 1], duration: 600 }, '-=200');
  },

  zmq(slide) {
    const tl = createTimeline()
      .add(slide.querySelectorAll('.boite'), {
        scale: [0.7, 1], opacity: [0, 1], delay: stagger(150), duration: 500, ease: 'outBack',
      });
    // Messages en boucle : capteurs vers la droite, commandes vers la gauche.
    const kfOpacity = [
      { to: 1, duration: 150 }, { to: 1, duration: 1100 }, { to: 0, duration: 350 },
    ];
    tl.add(slide.querySelectorAll('.flux-capteurs .msg'), {
      left: ['0%', '96%'], opacity: kfOpacity,
      delay: stagger(550), duration: 1600, loop: true, ease: 'linear',
    }, '-=100');
    tl.add(slide.querySelectorAll('.flux-commandes .msg'), {
      left: ['96%', '0%'], opacity: kfOpacity,
      delay: stagger(550), duration: 1600, loop: true, ease: 'linear',
    }, '<<');
    return tl;
  },

  monde(slide) {
    return createTimeline()
      .add(slide.querySelectorAll('.tuile'), {
        translateY: ['30px', 0], opacity: [0, 1],
        delay: stagger(140), duration: 550, ease: 'outCubic',
      })
      .add(slide.querySelector('.ligne-bonus'), { opacity: [0, 1], duration: 500 });
  },

  percevoir(slide) {
    return createTimeline()
      .add(slide.querySelector('.svg-cam'), { opacity: [0, 1], duration: 500 })
      .add(slide.querySelector('.fleche'), { opacity: [0, 1], translateX: ['-15px', 0], duration: 400 })
      .add(slide.querySelector('.svg-mask'), { opacity: [0, 1], duration: 500 })
      .add(slide.querySelector('.honnete'), { opacity: [0, 1], translateY: ['15px', 0], duration: 600 }, '+=300');
  },

  conduire(slide) {
    const el = slide.querySelector('#compteur-circuits');
    const obj = { n: 0 };
    return animate(obj, {
      n: 30, duration: 1600, ease: 'outExpo',
      modifier: utils.round(0),
      onUpdate: () => { el.textContent = obj.n; },
    });
  },

  circuits(slide) {
    return animate(slide.querySelectorAll('.galerie-circuits img'), {
      opacity: [0, 1], scale: [0.6, 1],
      delay: stagger(110), duration: 550, ease: 'outBack',
    });
  },

  panneaux(slide) {
    const acc = slide.querySelector('#compteur-acc');
    const obj = { v: 0 };
    return createTimeline()
      .add(slide.querySelectorAll('.etape'), {
        opacity: [0, 1], translateY: ['20px', 0],
        delay: stagger(220), duration: 450, ease: 'outCubic',
      })
      .add(slide.querySelectorAll('.pipe-fleche'), {
        opacity: [0, 1], delay: stagger(220), duration: 300,
      }, '-=1000')
      .add(obj, {
        v: 0.9513, duration: 1200, ease: 'outExpo',
        onUpdate: () => { acc.textContent = obj.v.toFixed(4).replace('.', ','); },
      });
  },

  occlusion(slide) {
    const sign = slide.querySelector('#sign-occlusion');
    const pred = slide.querySelector('#pred-occlusion');
    const setPred = (txt, ok) => {
      pred.innerHTML = 'panneau lu : <strong>' + txt + '</strong>';
      pred.classList.toggle('pred-ko', ok === false);
      pred.classList.toggle('pred-ok', ok === true);
    };
    setPred('—', null);
    return createTimeline({ loop: true, loopDelay: 2200, onLoop: () => setPred('—', null) })
      // Le panneau glisse vers le bord gauche du champ : le « 3 » sort,
      // il ne reste qu'un « 0 » visible — d'où la lecture « 90 » à conf 1.00.
      .add(sign, { left: ['110px', '-58px'], duration: 1500, ease: 'inOutQuad' })
      .call(() => setPred('90 · conf 1.00', false), '-=100')
      // Le fix : placement replafonné à 46 px, panneau entièrement visible.
      .add(sign, { left: '115px', duration: 900, ease: 'outCubic' }, '+=1700')
      .call(() => setPred('30 · conf 0.98', true), '-=50');
  },

  mesurer(slide) {
    // Barres diagnostic « urgence » horizontales : on anime la LARGEUR (% de
    // la cellule), pas la hauteur — le layout est passé en histogramme couché.
    return animate(slide.querySelectorAll('.barres-urgence .barre'), {
      width: (el) => ['0%', el.dataset.h + '%'],
      delay: stagger(200), duration: 900, ease: 'outCubic',
    });
  },

  resultats(slide) {
    // Les barres CSS ont été remplacées par les deux graphes matplotlib réels ;
    // on anime leur apparition (fondu + montée) pour garder le mouvement.
    return animate(slide.querySelectorAll('.result-graph'), {
      opacity: [0, 1], translateY: ['26px', 0],
      delay: stagger(180), duration: 700, ease: 'outCubic',
    });
  },

  conclusion(slide) {
    const el = slide.querySelector('#compteur-vitesse');
    const obj = { v: 70.6 };
    return animate(obj, {
      v: 100.0, duration: 2200, ease: 'inOutQuad', delay: 600,
      onUpdate: () => { el.textContent = obj.v.toFixed(1).replace('.', ','); },
    });
  },
};

let animEnCours = null;

function jouer(slide) {
  if (animEnCours) { animEnCours.revert(); animEnCours = null; }
  LiveDrive.stop(); // coupe le pilote live dès qu'on quitte sa slide
  const nom = slide && slide.dataset.anim;
  if (nom === 'demo-live') {
    LiveDrive.start(
      slide.querySelector('#live-canvas'),
      'gen_014',
      { hud: slide.querySelector('#live-hud') },
    );
    return;
  }
  if (nom && ANIMS[nom]) animEnCours = ANIMS[nom](slide);
}

/* Rendu statique : pas de timeline. La classe .statique (+ CSS) révèle les
   états initiaux opacity:0 ; on pose ici les valeurs que le CSS ne peut pas
   déduire — hauteurs de barres (même formule que les timelines) et compteurs
   à leur valeur finale, pour un export PDF complet. */
function finaliserStatique() {
  document.documentElement.classList.add('statique');
  // Barres « urgence » horizontales : largeur = % (pas de hauteur). Les graphes
  // .result-graph sont révélés par le CSS .statique (opacity:1).
  document.querySelectorAll('.barres-urgence .barre[data-h]').forEach((b) => {
    b.style.width = b.dataset.h + '%';
  });
  const poser = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  poser('compteur-circuits', '30');
  poser('compteur-acc', '0,9513');
  poser('compteur-vitesse', '100,0');
  const pred = document.getElementById('pred-occlusion');
  if (pred) pred.innerHTML = 'panneau lu : <strong>30 · conf 0.98</strong>';
}

Reveal.initialize({
  width: 1280,
  height: 720,
  hash: true,
  transition: 'fade',
  // 14 colonnes : 13 slides + la pile d'annexes (les verticales ne comptent pas).
  slideNumber: () => [Reveal.getIndices().h + 1, '/', 14],
  controls: true,
  progress: true,
  plugins: [RevealNotes],
}).then(() => {
  if (STATIQUE) { finaliserStatique(); return; }
  jouer(Reveal.getCurrentSlide());
  Reveal.on('slidechanged', (e) => jouer(e.currentSlide));
});
