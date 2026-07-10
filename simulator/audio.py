"""
audio.py — Sons de course procéduraux (aucun fichier .wav requis).

Moteur : DEUX boucles continues (un grave + un aigu) jouees en permanence,
dont on module seulement le VOLUME selon la vitesse. Comme les boucles ne
redemarrent jamais, il n'y a aucun clic ; l'oreille entend le moteur
"s'ouvrir" quand la vitesse monte (l'aigu prend le dessus). Ondes sinus +
harmoniques douces (pas de dent-de-scie/carre qui buzz).

- Frein : leger crissement de pneus (bruit filtre, volume modere).
- Impact : choc court quand la voiture percute le mur.

Si aucun peripherique audio n'est dispo (headless), l'init echoue proprement
et toutes les methodes deviennent des no-op. Aucun impact sur la simulation.
"""

import numpy as np

try:
    import pygame
except ImportError:  # pragma: no cover
    pygame = None


SR = 44100          # frequence d'echantillonnage
MUSIC_VOL = 0.07    # volume de la musique de fond : TRES discret


class EngineAudio:
    """Son moteur (2 boucles continues) + musique + effets. No-op si pas d'audio."""

    def __init__(self) -> None:
        self.enabled = False
        self.muted = False
        try:
            pygame.mixer.init(frequency=SR, size=-16, channels=2, buffer=512)
            self._low_ch = pygame.mixer.Channel(0)
            self._high_ch = pygame.mixer.Channel(1)
            self._brake_ch = pygame.mixer.Channel(2)
            self._impact_ch = pygame.mixer.Channel(3)
            self._music_ch = pygame.mixer.Channel(4)

            # Deux nappes moteur continues (jamais redemarrees).
            low = self._make_rumble(base=64.0, harmonics=(1.0, 0.5, 0.25), tremolo=7.0)
            high = self._make_rumble(base=132.0, harmonics=(1.0, 0.4, 0.2), tremolo=11.0)
            self._brake_snd = self._make_screech()
            self._impact_snd = self._make_impact()
            music = self._make_music()

            self._low_ch.play(low, loops=-1)
            self._high_ch.play(high, loops=-1)
            self._music_ch.play(music, loops=-1)
            self._low_ch.set_volume(0.0)
            self._high_ch.set_volume(0.0)
            self._music_ch.set_volume(MUSIC_VOL)
            self.enabled = True
        except Exception as e:  # mixer indispo (headless) -> no-op silencieux
            print(f"[Audio] desactive (pas de peripherique audio) : {e}")

    # --- Generation des sons ---------------------------------------------

    def _make_rumble(self, base: float, harmonics, tremolo: float, dur: float = 1.0):
        """Nappe moteur douce : fondamentale sinus + harmoniques faibles,
        legere modulation d'amplitude (tremolo) pour donner du 'grain'.
        Duree = nombre entier de periodes de la fondamentale -> boucle nette."""
        n_period = max(1, int(SR / base))
        reps = max(1, int(SR * dur / n_period))
        n = n_period * reps
        t = np.arange(n)
        wave = np.zeros(n, dtype=np.float64)
        for k, amp in enumerate(harmonics, start=1):
            wave += amp * np.sin(2 * np.pi * base * k * t / SR)
        wave /= sum(harmonics)
        # Tremolo : modulation d'amplitude lente (nombre entier de cycles).
        trem_cycles = max(1, round(tremolo * dur))
        trem = 0.85 + 0.15 * np.sin(2 * np.pi * trem_cycles * t / n)
        return self._to_sound(wave * trem * 0.9)

    def _make_screech(self, dur: float = 0.45):
        n = int(SR * dur)
        noise = np.random.uniform(-1, 1, n)
        k = 12                                   # passe-bas -> moins agressif
        noise = np.convolve(noise, np.ones(k) / k, mode="same")
        env = np.linspace(1.0, 0.0, n) ** 1.6
        return self._to_sound(noise * env * 0.28)

    def _make_impact(self, dur: float = 0.3):
        n = int(SR * dur)
        t = np.arange(n)
        thud = np.sin(2 * np.pi * 70 * t / SR) * np.exp(-t / (0.08 * SR))
        crack = np.random.uniform(-1, 1, n) * np.exp(-t / (0.04 * SR))
        return self._to_sound((0.7 * thud + 0.5 * crack) * 0.55)

    def _make_music(self, bpm: int = 124):
        """Boucle de fond ~8 s, douce, feeling 'course' : basse + arpege sur
        une progression d'accords (Am - F - C - G). Ondes sinus + harmonique
        legere, enveloppe douce. Volume final regle bas via le canal."""
        beat = SR * 60 // bpm            # echantillons par temps
        bar = beat * 4                   # 4 temps par mesure
        prog = [220.0, 174.61, 261.63, 196.0]   # A3, F3, C4, G3 (fondamentales)
        total = bar * len(prog)
        buf = np.zeros(total, dtype=np.float64)

        def note(freq, start, dur, amp):
            n = int(dur)
            t = np.arange(n)
            attack = np.minimum(1.0, t / (0.015 * SR))
            decay = np.exp(-t / (dur * 0.6))
            env = attack * decay
            w = np.sin(2 * np.pi * freq * t / SR) + 0.25 * np.sin(2 * np.pi * 2 * freq * t / SR)
            end = min(start + n, total)
            buf[start:end] += (w * env * amp)[:end - start]

        half = beat // 2
        for bi, root in enumerate(prog):
            base = bi * bar
            # Basse : root une octave plus bas, une note par temps.
            for b in range(4):
                note(root / 2, base + b * beat, int(beat * 0.9), 0.16)
            # Arpege en croches : root, quinte, octave, quinte.
            arp = [root, root * 1.5, root * 2.0, root * 1.5]
            for j in range(8):
                note(arp[j % 4], base + j * half, int(half * 0.8), 0.09)

        return self._to_sound(np.clip(buf, -1, 1))

    @staticmethod
    def _to_sound(buf: np.ndarray):
        audio = (np.clip(buf, -1, 1) * 32767).astype(np.int16)
        stereo = np.ascontiguousarray(np.column_stack([audio, audio]))
        return pygame.sndarray.make_sound(stereo)

    # --- API ------------------------------------------------------------

    def update_engine(self, speed_px: float, max_speed_px: float) -> None:
        if not self.enabled or self.muted:
            return
        ratio = 0.0 if max_speed_px <= 0 else max(0.0, min(1.0, speed_px / max_speed_px))
        # Grave : toujours present (ralenti), monte un peu. Aigu : monte fort.
        self._low_ch.set_volume(0.12 + 0.10 * ratio)
        self._high_ch.set_volume(0.02 + 0.28 * ratio)

    def play_brake(self, speed_px: float) -> None:
        if not self.enabled or self.muted:
            return
        if speed_px > 55 and not self._brake_ch.get_busy():
            self._brake_ch.set_volume(min(0.6, speed_px / 300.0))
            self._brake_ch.play(self._brake_snd)

    def play_impact(self) -> None:
        if not self.enabled or self.muted:
            return
        self._impact_ch.play(self._impact_snd)

    def toggle_mute(self) -> bool:
        self.muted = not self.muted
        if self.enabled:
            if self.muted:
                self._low_ch.set_volume(0.0)
                self._high_ch.set_volume(0.0)
                self._music_ch.set_volume(0.0)
            else:
                # le moteur sera reajuste au prochain update_engine ;
                # la musique doit etre restauree explicitement.
                self._music_ch.set_volume(MUSIC_VOL)
        return self.muted

    def close(self) -> None:
        if self.enabled:
            try:
                pygame.mixer.stop()
                pygame.mixer.quit()
            except Exception:
                pass
