/* live-drive.js — le pilote PID du projet, porté en JavaScript, qui conduit
   EN DIRECT dans une slide du deck, sur les vrais PNG de circuits.

   Portage fidèle constante par constante :
   - simulator/physics.py  : bicycle (LF=LR=8.5), MAX_ACCELERATION=170,
     BRAKE_FORCE=200, FRICTION=0.02 PAR FRAME (non scalée dt, comme
     l'original), MAX_SPEED_PX=300, SPEED_SCALE=0.72.
   - simulator/sensors.py  : lidar 5 rayons (-60,-30,0,30,60°), portée
     300 px, pas de 1 px, arrêt au premier pixel non-route (trait noir).
   - pilot/control.py      : pid_policy — gains KP 0.25/0.12, front-turn
     (seuil 120, gain 0.8), tight boost (40 px, +35°), clip ±45°,
     amortissement 0.5, throttle interpolé 40→70 px, soft cap vitesse.
   La boucle tourne à dt fixe 1/60 via accumulateur : mêmes conditions que
   le banc bench_laps.py, quel que soit le refresh de l'écran. */

(function () {
  'use strict';

  /* ---- Constantes physiques (simulator/physics.py) ---- */
  const LF = 8.5, LR = 8.5;
  const MAX_ACCELERATION = 170.0, BRAKE_FORCE = 200.0;
  const FRICTION = 0.02, MAX_SPEED_PX = 300.0, SPEED_SCALE = 0.72;
  const DT = 1 / 60;

  /* ---- Lidar (simulator/sensors.py) ---- */
  const LIDAR_ANGLES = [-60, -30, 0, 30, 60].map((a) => a * Math.PI / 180);
  const LIDAR_MAX_RANGE = 300;

  /* ---- PID (pilot/control.py) ---- */
  const KP_LAT_CLOSE = 0.25, KP_LAT_FAR = 0.12;
  const FRONT_TURN_THRESHOLD = 120.0, FRONT_TURN_GAIN = 0.8;
  const TIGHT_THRESHOLD = 40.0, TIGHT_BOOST = 35.0;
  const THROTTLE_MAX = 1.0, THROTTLE_MIN = 0.25;
  const LIDAR_FRONT_SAFE = 70.0, LIDAR_FRONT_SLOW = 40.0;
  const SPEED_TARGET_DEFAULT = 120.0;
  const STEER_DAMPING = 0.5;

  /* Positions de départ (assets/tracks/<nom>.json du repo, seed 1000). */
  const STARTS = {
    gen_000: { x: 845.7, y: 342.4, a: -89.74 },
    gen_003: { x: 963.6, y: 348.9, a: -70.36 },
    gen_007: { x: 911.1, y: 366.1, a: -53.67 },
    gen_014: { x: 904.1, y: 371.5, a: -24.82 },
    gen_021: { x: 897.6, y: 368.7, a: -89.75 },
    gen_026: { x: 918.5, y: 368.5, a: -89.72 },
  };

  let rafId = null;
  let etat = null;

  function chargerCircuit(nom) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const off = document.createElement('canvas');
        off.width = img.naturalWidth;
        off.height = img.naturalHeight;
        const c = off.getContext('2d', { willReadFrequently: true });
        c.drawImage(img, 0, 0);
        const data = c.getImageData(0, 0, off.width, off.height);
        resolve({ img, data, w: off.width, h: off.height });
      };
      img.onerror = reject;
      img.src = 'assets/circuits/' + nom + '.png';
    });
  }

  function estRoute(t, x, y) {
    const px = x | 0, py = y | 0;
    if (px < 0 || py < 0 || px >= t.w || py >= t.h) return false;
    const i = (py * t.w + px) * 4;
    const d = t.data.data;
    return (d[i] + d[i + 1] + d[i + 2]) / 3 > 200; // blanc = route, trait noir = bord
  }

  function lidar(t, car) {
    const out = [];
    for (const off of LIDAR_ANGLES) {
      const a = car.angle + off;
      const ca = Math.cos(a), sa = Math.sin(a);
      let dist = LIDAR_MAX_RANGE;
      for (let d = 1; d <= LIDAR_MAX_RANGE; d++) {
        if (!estRoute(t, car.x + d * ca, car.y + d * sa)) { dist = d; break; }
      }
      out.push(dist);
    }
    return out;
  }

  function pid(l, speedKmh, st) {
    const [leftFar, leftClose, front, rightClose, rightFar] = l;
    let steering = KP_LAT_CLOSE * (rightClose - leftClose) + KP_LAT_FAR * (rightFar - leftFar);
    if (front < FRONT_TURN_THRESHOLD) {
      const urgency = 1.0 - front / FRONT_TURN_THRESHOLD;
      steering += FRONT_TURN_GAIN * ((rightClose + rightFar) - (leftClose + leftFar)) * urgency / 10.0;
    }
    if (leftClose < TIGHT_THRESHOLD && leftClose < rightClose) steering += TIGHT_BOOST;
    else if (rightClose < TIGHT_THRESHOLD && rightClose < leftClose) steering -= TIGHT_BOOST;
    steering = Math.max(-45, Math.min(45, steering));
    steering = STEER_DAMPING * st.prevSteering + (1 - STEER_DAMPING) * steering;
    st.prevSteering = steering;

    let tt = (front - LIDAR_FRONT_SLOW) / (LIDAR_FRONT_SAFE - LIDAR_FRONT_SLOW);
    tt = Math.max(0, Math.min(1, tt));
    let throttle = THROTTLE_MIN + tt * (THROTTLE_MAX - THROTTLE_MIN);
    if (speedKmh > st.speedTarget) throttle = Math.min(throttle, 0.4);
    return { steering, throttle };
  }

  function physique(car, cmd) {
    if (cmd.throttle > 0) car.speed += cmd.throttle * MAX_ACCELERATION * DT;
    car.speed -= car.speed * FRICTION; // par frame, comme physics.py
    car.speed = Math.max(0, Math.min(MAX_SPEED_PX, car.speed));
    const delta = cmd.steering * Math.PI / 180;
    const beta = Math.atan2(LR * Math.tan(delta), LF + LR);
    car.x += car.speed * Math.cos(car.angle + beta) * DT;
    car.y += car.speed * Math.sin(car.angle + beta) * DT;
    car.angle += (car.speed / LR) * Math.sin(beta) * DT;
  }

  function resetVoiture(st) {
    const s = STARTS[st.nom];
    st.car = { x: s.x, y: s.y, angle: s.a * Math.PI / 180, speed: 0 };
    st.prevSteering = 0;
    st.horsPiste = 0;
    st.loin = false;
    st.tour = 0;
    st.chrono = 0;
    st.dernierTour = null;
  }

  function tick(st) {
    const l = lidar(st.track, st.car);
    const kmh = st.car.speed * SPEED_SCALE;
    const cmd = pid(l, kmh, st);
    physique(st.car, cmd);
    st.chrono += DT;
    st.dernierLidar = l;
    st.dernierCmd = cmd;

    // Compteur de tours : on s'éloigne du départ puis on y revient.
    const s = STARTS[st.nom];
    const dStart = Math.hypot(st.car.x - s.x, st.car.y - s.y);
    if (dStart > 200) st.loin = true;
    if (st.loin && dStart < 40) {
      st.tour += 1;
      st.dernierTour = st.chrono;
      st.chrono = 0;
      st.loin = false;
    }

    // Filet démo : si la voiture sort vraiment (1 s hors piste), on respawn.
    if (!estRoute(st.track, st.car.x, st.car.y)) {
      if (++st.horsPiste > 60) resetVoiture(st);
    } else {
      st.horsPiste = 0;
    }
  }

  function dessiner(st) {
    const ctx = st.ctx, t = st.track, car = st.car;
    const sx = st.canvas.width / t.w, sy = st.canvas.height / t.h;
    ctx.clearRect(0, 0, st.canvas.width, st.canvas.height);
    ctx.drawImage(t.img, 0, 0, st.canvas.width, st.canvas.height);

    // Rayons lidar (vert = marge, rouge = court)
    ctx.save();
    ctx.lineWidth = 1.5;
    st.dernierLidar.forEach((d, i) => {
      const a = car.angle + LIDAR_ANGLES[i];
      ctx.strokeStyle = d < 70 ? 'rgba(239,68,68,.85)' : 'rgba(34,197,94,.55)';
      ctx.beginPath();
      ctx.moveTo(car.x * sx, car.y * sy);
      ctx.lineTo((car.x + d * Math.cos(a)) * sx, (car.y + d * Math.sin(a)) * sy);
      ctx.stroke();
    });

    // Voiture
    ctx.translate(car.x * sx, car.y * sy);
    ctx.rotate(car.angle);
    ctx.fillStyle = '#ef4444';
    ctx.fillRect(-9, -5, 18, 10);
    ctx.fillStyle = '#0b1220';
    ctx.fillRect(2, -4, 5, 8); // pare-brise
    ctx.restore();

    // HUD minimal (le style fin est laissé au thème du deck)
    const kmh = (car.speed * SPEED_SCALE).toFixed(0);
    const hud = st.hudEl;
    if (hud) {
      hud.textContent = kmh + ' km/h · volant ' + st.dernierCmd.steering.toFixed(1) +
        '° · tour ' + st.tour + (st.dernierTour ? ' · dernier ' + st.dernierTour.toFixed(1) + ' s' : '');
    }
  }

  function boucle(ts) {
    const st = etat;
    if (!st) return;
    if (st.tsPrec == null) st.tsPrec = ts;
    st.acc += Math.min((ts - st.tsPrec) / 1000, 0.1);
    st.tsPrec = ts;
    while (st.acc >= DT) { tick(st); st.acc -= DT; }
    dessiner(st);
    rafId = requestAnimationFrame(boucle);
  }

  window.LiveDrive = {
    async start(canvas, nom, opts) {
      this.stop();
      const track = await chargerCircuit(nom);
      canvas.width = canvas.width || track.w;
      canvas.height = canvas.height || track.h;
      etat = {
        canvas, nom, track,
        ctx: canvas.getContext('2d'),
        hudEl: opts && opts.hud ? opts.hud : null,
        speedTarget: (opts && opts.speedTarget) || SPEED_TARGET_DEFAULT,
        acc: 0, tsPrec: null, dernierLidar: [0, 0, 0, 0, 0], dernierCmd: { steering: 0, throttle: 0 },
      };
      resetVoiture(etat);
      rafId = requestAnimationFrame(boucle);
    },
    stop() {
      if (rafId) cancelAnimationFrame(rafId);
      rafId = null;
      etat = null;
    },
    circuits: Object.keys(STARTS),
  };
})();
